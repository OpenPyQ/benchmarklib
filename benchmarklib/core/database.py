"""
Quantum Benchmarking Database Library (Simplified Single-Problem Design)

A unified database interface for quantum circuit benchmarking, designed for
single problem types per database. Each problem type (3SAT, Clique, etc.)
should have its own directory with a dedicated database.

Database Schema (per problem type):
- problem_instances: Stores unique problem instances of one type
- trials: Stores trial results referencing problem instances by ID

Key Features:
- Single problem type per database (simplified design)
- Normalized database (no duplicate problem storage across trials)
- Abstract oracle method for centralized circuit generation
- Async job result fetching from IBM Quantum
- Comprehensive documentation and maintenance tools

"""
from abc import ABC, abstractmethod
import asyncio
import io
import json
import logging
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type
from sqlalchemy import Column, Integer, String, JSON, Float, DateTime, LargeBinary, UniqueConstraint, select, create_engine, select, func, or_, text, update
from sqlalchemy.orm import declarative_base, Mapped, relationship, mapped_column, sessionmaker, Session, DeclarativeBase, Mapped, relationship, selectinload, joinedload
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import DetachedInstanceError
from qiskit import QuantumCircuit, qpy
from qiskit_ibm_runtime import RuntimeJobFailureError
from qiskit.providers import Backend

from .types import Base, BaseTrial, BaseProblem
from benchmarklib.pipeline.config import PipelineConfig

# Configure logging
logger = logging.getLogger("benchmarklib.core.database")


class BackendProperty(Base):
    __tablename__ = "backend_properties"

    backend_name: Mapped[str]
    last_update_date: Mapped[datetime]
    properties: Mapped[Dict[str, Any]] = mapped_column(MutableDict.as_mutable(JSON))
    
    # backend_name + last_update_date should be unique together
    __table_args__ = (
        UniqueConstraint("backend_name", "last_update_date", name="uq_backend_name_last_update"),
    )

    def __repr__(self) -> str:
        return f"<BackendProperty(id={self.id}, backend_name={self.backend_name})>"
    
    def get_gate_errors(self) -> Dict[str, List[float]]:
        error_rates = {}
        for d in self.properties.get("gates", []):
            if d["gate"] not in error_rates:
                error_rates[d["gate"]] = []

            for param in d["parameters"]:
                if param["name"] == "gate_error":
                    error_rates[d["gate"]].append(param["value"])

        return error_rates

    def get_average_gate_errors(self) -> Dict[str, np.floating[Any]]:
        return {g: np.mean(v) for g, v in self.get_gate_errors().items()}

class DatabaseManager:
    """
    Manager for general database connections.
    """
    def __init__(
        self,
        db_name: str,
    ):
        """
        Args:
            db_name: SQLite database filename
        """
        self.db_name = db_name

        self.engine = create_engine(
            f"sqlite:///{db_name}",
            connect_args={
                "timeout": 30.0,
                "check_same_thread": False,
            },
            echo=False,  # Set to True for SQL debugging
        )

        # configure database connections to support concurrency, balance safety/speed, and enable foreign keys
        with self.engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.commit()
        
        # specify defaults for sessions with a factory
        self.Session = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False,  # we often work with objects detached from sessions
        )

        self.AsyncSession = sessionmaker(
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False,
        )
        
        Base.metadata.create_all(bind=self.engine)
    

    def session(self) -> Session:
        """
        Create a new database session for direct queries.
        
        Returns:
            SQLAlchemy Session object (use as context manager)
            
        Example:
            with db.session() as session:
                problems = session.query(db.problem_class).all()
                session.commit()
        """
        return self.Session()
    
    def async_session(self) -> AsyncSession:
        """
        Create a new async database session for direct queries.

        Returns:
            SQLAlchemy AsyncSession object (use as context manager)

        Example:
            with db.async_session() as session:
                query = select(db.problem_class).where(complexity >= 10)
                problems = await session.execute(query)
                session.commit()
        """
        return self.AsyncSession()
    
    def query(self, query) -> List[Any]:
        """
        Execute a query without managing a session.

        Args:
            query: SQLAlchemy query object (e.g., select(...).where(...))

        Returns:
            List of results
        """
        with self.session() as session:
            results = session.execute(query).scalars().all()
            return results

    def get_or_create(self, model: Type[Base], defaults: Dict[str, Any]={}, **kwargs):
        """
        Get or create a database record.

        Args:
            model: SQLAlchemy model class
            defaults: Default values for creation
            **kwargs: Filter criteria
        """
        with self.session() as session:
            instance = session.execute(
                select(model).filter_by(**kwargs)
            ).scalars().first()
            if instance:
                session.expunge(instance)
                return instance
            else:
                params = {**kwargs, **defaults}
                instance = model(**params)
                session.add(instance)
                session.commit()
                session.refresh(instance)
                session.expunge(instance)
                return instance
        

class BackendPropertyManager(DatabaseManager):
    def latest(self, backend: Backend | str, as_of: Optional[datetime] = None) -> Optional[BackendProperty]:
        """
        Get the latest backend properties as of a specific date.
        Args:
            backend: Backend object or name
        Args:
            as_of: Date to filter properties (None for latest overall)
        """
        backend_name = backend.name if isinstance(backend, Backend) else backend
        with self.session() as session:
            query = select(BackendProperty).where(BackendProperty.backend_name == backend_name).order_by(BackendProperty.last_update_date.desc())
            if as_of:
                query = query.where(BackendProperty.last_update_date <= as_of)
            result = session.execute(query).scalars().first()
            return result
        
    def load_missing_dates(self, backend: Backend, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> None:
        """
        Load missing backend properties for days without data.

        Args:
            backend: Backend to load properties for
            start_date: Start date for loading (None for earliest available)
            end_date: End date for loading (None for latest available)
        """
        # identify days we already have data for
        existing_dates = self.query(
            select(BackendProperty.last_update_date).distinct().where(
                BackendProperty.backend_name == backend.name
            )
        )
        existing_dates = {dt.date() for dt in existing_dates}

        # determine date range to check
        if start_date is None:
            start_date = min(existing_dates) if existing_dates else datetime.utcnow().date()
        if end_date is None:
            end_date = datetime.utcnow().date()
        date = datetime.combine(start_date, datetime.max.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        with self.session() as session:
            while date <= end_date:
                if date.date() not in existing_dates:
                    # load properties for this date
                    try:
                        backend_props = backend.properties(datetime=date)
                        if backend_props.last_update_date.date() in existing_dates:
                            date += timedelta(days=1)
                            continue
                        bp = BackendProperty(
                            backend_name=backend.name,
                            last_update_date=backend_props.last_update_date,
                            properties=json.loads(json.dumps(backend_props.to_dict(), default=str)),  # dump and load to ensure JSON serializability
                        )
                        session.add(bp)
                        session.commit()
                        existing_dates.add(date.date())
                        logger.info(f"Saved backend properties for {backend.name} on {date}")
                    except Exception as e:
                        logger.error(f"Failed to load/save properties for {backend.name} on {date}: {e}")
                date += timedelta(days=1)

class BenchmarkDatabase(DatabaseManager):
    """
    SQLAlchemy-based Database manager for a single quantum problem type benchmark.

    This class manages both problem instances and trials for ONE problem type
    in a normalized database schema. Each problem type should have its own
    database instance and file.

    Type Safety:
    The database expects to work with consistent BaseProblem and BaseTrial subclass tables.
    Register these types when creating the database.

    Usage:
        # Initialize with problem and trial classes
        db = BenchmarkDatabase(
            db_name="rbf.db",
            problem_class=RandomBooleanFunction,
            trial_class=RandomBooleanFunctionTrial
        )
        
        # high-level methods (backward compatible with previous database manager)
        db.save_problem_instance(
            problem = RandomBooleanFunction(...)
        )
        trials = db.find_trials(instance_id=1)

        # medium-level approach (for custom sqlalchemy queries that don't require further use of a session)
        trials = db.query(select(db.trial_class).join(db.problem_class).where(
            db.trial_class.compiler_name == "XAG",
            db.problem_class.num_vars >= 5
        ))

        # another example with shortform syntax for simple filtering
        problems = db.query(db.problems.filter_by(num_vars=5))
        
        # low-level approach (use direct session access for custom queries)
        with db.session() as session:
            problems = session.scalars(db.problems.filter_by(num_vars=5))
    """

    def __init__(
        self,
        db_name: str,
        problem_class: Type[BaseProblem],
        trial_class: Type[BaseTrial],
        *args, **kwargs
    ):
        """
        Initialize database for a specific problem type.

        Args:
            db_name: SQLite database filename
            problem_class: BaseProblem subclass for this database
            trial_class: BaseTrial subclass for this database
        """
        super().__init__(db_name=db_name, *args, **kwargs)
        
        self.problem_class = problem_class
        self.trial_class = trial_class

        self.problem_type = problem_class.problem_type

        logger.info(f"Database initialized: {self.db_name} ({self.problem_type})")

    
    @property
    def problems(self):
        return select(self.problem_class)

    @property
    def trials(self):
        return select(self.trial_class)


    # Problem Instance Operations
    def save_problem_instance(self, problem: BaseProblem) -> int:
        """
        Save problem instance to database.

        Args:
            problem: Problem instance to save (must match registered type)

        Returns:
            Instance ID (sets problem.id as side effect)

        Raises:
            TypeError: If problem is not of the expected type
        """
        if not isinstance(problem, self.problem_class):
            raise TypeError(
                f"Expected {self.problem_class.__name__}, got {type(problem).__name__}"
            )

        with self.session() as session:
            if problem.id is None:
                # New problem instance
                session.add(problem)
                session.commit()
                session.refresh(problem)
                logger.debug(f"Saved new problem instance: {problem.id}")
            else:
                # Update existing
                problem.updated_at = datetime.utcnow()
                session.merge(problem)
                session.commit()
                logger.debug(f"Updated problem instance: {problem.id}")

            session.expunge(problem)
            
            return problem.id

    def get_problem_instance(self, instance_id: int) -> BaseProblem:
        """
        Retrieve problem instance by ID.

        Args:
            instance_id: Database ID of problem instance

        Returns:
            Problem instance object of the registered type

        Raises:
            ValueError: If instance not found
        """
        with self.session() as session:
            problem = session.get(self.problem_class, instance_id)
            if not problem:
                raise ValueError(f"Problem instance {instance_id} not found")
            
            # Detach from session for return
            session.expunge(problem)
            return problem

    def find_problem_instances(
        self,
        limit: Optional[int] = None,
        choose_untested: bool = False,
        random_sample: bool = False,
        compiler_name: Optional[str] = None,
        **filters: Optional[Dict[str, Any]],
    ) -> List[BaseProblem]:
        """
        Find problem instances matching criteria.

        Args:
            
            limit: Maximum number of results
            choose_untested: If True, only return problems with no trials (for specified compiler if provided)
            random_sample: If True, randomly sample from filtered results
            compiler_name: When choose_untested=True, find instances untested by this specific compiler
            **filters: Filter by problem attributes (e.g., num_vars=5)

        Returns:
            List of matching problem instances

        Example:
            problems = db.find_problem_instances(num_vars=5, limit=10)
        """

        with self.session() as session:
            query = select(self.problem_class)

            if choose_untested:
                if compiler_name:
                    # filter by problems with no trials for this specific compiler
                    subq = (
                        select(self.trial_class.problem_id)
                        .where(self.trial_class.compiler_name == compiler_name)
                        .distinct()
                    )
                    query = query.where(~self.problem_class.id.in_(subq))
                else:
                    # filter by problems with no trials at all
                    subq = select(self.trial_class.problem_id).distinct()
                    query = query.where(~self.problem_class.id.in_(subq))

            if filters:
                for key, value in filters.items():
                    column = getattr(self.problem_class, key, None)
                    if column is not None:
                        query = query.where(column == value)

            if random_sample:
                query = query.order_by(func.random())

            if limit:
                query = query.limit(limit)

            results = session.execute(query).scalars().all()
            return results

    def delete_problem_instance(self, instance_id: int) -> None:
        """
        Delete problem instance and cascade delete trials.
        
        Args:
            instance_id: ID to delete
        """
        with self.session() as session:
            problem = session.get(self.problem_class, instance_id)
            if problem:
                session.delete(problem)
                session.commit()
                logger.info(f"Deleted problem instance ID: {instance_id}")
            

    # Trial Operations
    def save_trial(self, trial: BaseTrial) -> int:
        """
        Save trial to database.

        Args:
            trial: Trial to save (must match registered type)

        Returns:
            Trial ID (sets trial.id as side effect)

        Raises:
            TypeError: If trial is not of the expected type
        """
        if not isinstance(trial, self.trial_class):
            raise TypeError(
                f"Expected {self.trial_class.__name__}, got {type(trial).__name__}"
            )

        with self.session() as session:
            if trial.id is None:
                session.add(trial)
                session.commit()
                session.refresh(trial)
                logger.debug(f"Saved new trial ID: {trial.id}")
            else:
                trial.updated_at = datetime.utcnow()
                session.merge(trial)
                session.commit()
                logger.debug(f"Updated trial ID: {trial.id}")

        return trial.id

    def get_trial(self, trial_id: int) -> BaseTrial:
        """
        Retrieve trial by ID.

        Args:
            trial_id: Database ID of trial

        Returns:
            Trial object of the registered type

        Raises:
            ValueError: If trial not found
        """
        with self.session() as session:
            # eagerly select the problem for the trial
            query = (
                select(self.trial_class)
                .where(self.trial_class.id == trial_id)
                .options(joinedload(self.trial_class.problem))
            )
            trial = session.execute(query).unique().scalar_one()
            if not trial:
                raise ValueError(f"Trial {trial_id} not found")
            
            # mark problem as used before expunging from session
            _ = trial.problem
            session.expunge_all()

            return trial

    def find_trials(
        self,
        trial_id: Optional[int] = None,
        instance_id: Optional[int] = None,
        job_id: Optional[str] = None,
        compiler_name: Optional[str] = None,
        include_pending: bool = True,
        limit: Optional[int] = None,
        **filters: Optional[Dict[str, Any]],
    ) -> List[BaseTrial]:
        """
        Find trials matching criteria.

        Args:
            trial_id: Specific trial ID
            instance_id: Filter by problem instance
            job_id: Filter by IBM Quantum job ID
            compiler_name: Filter by compilation method
            include_pending: Include trials without results
            filters: Filter by trial or problem attributes
            limit: Maximum number of results

        Returns:
            List of matching trials
        """
        
        with self.session() as session:
            query = select(self.trial_class).join(self.problem_class)

            if trial_id is not None:
                query = query.where(self.trial_class.id == trial_id)

            if instance_id is not None:
                query = query.where(self.trial_class.problem_id == instance_id)

            if job_id is not None:
                query = query.where(self.trial_class.job_id == job_id)

            if compiler_name is not None:
                query = query.where(self.trial_class.compiler_name == compiler_name)

            if not include_pending:
                query = query.where(self.trial_class.counts != None).where(self.trial_class.is_failed == False)

            if filters:
                for key, value in filters.items():
                    # apply where constraint to the trial or to its problem, depending on whether the 
                    # kwarg is part of the trial or problem class
                    if hasattr(self.trial_class, key):
                        query = query.where(getattr(self.trial_class, key) == value)
                    elif hasattr(self.problem_class, key):
                        query = query.where(getattr(self.problem_class, key) == value)

            if limit:
                query = query.limit(limit)

            # load related problem in the same query (for Many to One from trial_class to problem_class)
            query = query.options(joinedload(self.trial_class.problem))

            results = session.execute(query).scalars().all()
            session.expunge_all()

            return results

    def delete_trial(self, trial_id: int) -> None:
        """Delete trial from database."""
        with self.session() as session:
            trial = session.get(self.trial_class, trial_id)
            if trial:
                session.delete(trial)
                session.commit()
                logger.info(f"Deleted trial ID: {trial_id}")
            else:
                logger.warning(f"Unable to delete trial {trial}. Not found in database.")

    def create_compilation_failure(self, problem: BaseProblem, compiler_name: str) -> BaseTrial:
        trial = self.trial_class(
            problem=problem,
            compiler_name=compiler_name,
            counts=None,
            is_failed=True
        )
        self.save_trial(trial)
        return trial

    def get_saved_config(self, config: PipelineConfig) -> PipelineConfig:
        """
        Get or creates the PipelineConfig in the database, returning the object stored there.
        Use this to save a config to a trial, to avoid duplicating identical configs.
        Note: PipelineConfig requires a unique name; if a matching name is found, that object is returned. 
        This throws a warning if the config parameters differ for the matching name object.
        """
        with self.session() as session:
            query = select(PipelineConfig).where(
                PipelineConfig.name == config.name
            )
            existing = session.execute(query).scalars().first()
            if existing:
                if existing != config:
                    logger.warning(
                        f"PipelineConfig with name {config.name} already exists with different parameters."
                    )
                session.expunge(existing)
                return existing
            else:
                session.add(config)
                session.commit()
                session.refresh(config)
                session.expunge(config)
                logger.info(f"Saved new PipelineConfig with name {config.name}.")
                return config
            
    # Async Job Management
    def get_pending_job_ids(self) -> List[str]:
        """Get all job IDs with pending results."""
        with self.session() as session:
            query = (
                select(self.trial_class.job_id)
                .where(
                    self.trial_class.job_id != None,
                    self.trial_class.counts == None,
                    self.trial_class.is_failed == False,
                )
                .distinct()
                .order_by(self.trial_class.created_at)
            )
            results = session.execute(query).scalars().all()
            return list(results)

    async def update_job_results(self, job_id: str, service, result_register: Optional[str] = None) -> None:
        """
        Fetch and update results for a specific job.

        Args:
            job_id: IBM Quantum job ID
            service: QiskitRuntimeService instance
            result_register: (Optional) "c", "meas", etc. to specify which register to extract counts from. Default: check "c", then "meas".
        """

        try:
            # Fetch job results asynchronously
            logger.info(f"Fetching results for job {job_id}")
            retrieved_job = await asyncio.to_thread(service.job, job_id)
            results = await asyncio.to_thread(retrieved_job.result)
            
        except RuntimeJobFailureError:
            # Mark trials as failed
            logger.info(f"Job {job_id} failed; marking trials as failed")

            with self.session() as session:
                session.execute(
                        update(self.trial_class).where(
                        self.trial_class.job_id == job_id,
                        self.trial_class.counts == None,
                        self.trial_class.is_failed == False,
                    ).values(is_failed=True)
                )

                session.commit()
            return
        
        except Exception as e:
            logger.error(f"Error fetching job {job_id}: {e}")
            return
        
        try:
            # Update pending trials for this job
            with self.session() as session:
                trials = session.execute(
                    select(
                        self.trial_class.id,
                        self.trial_class.job_pub_idx,
                    ).where(
                        self.trial_class.job_id == job_id,
                        self.trial_class.counts == None,
                        self.trial_class.is_failed == False,
                    )
                ).fetchall()
                        
            updates = []
            for trial_id, job_pub_idx in trials:
                try:
                    # Extract counts from results
                    pub_result = results[job_pub_idx]

                    # Handle different result data structures
                    if result_register is not None:
                        counts = getattr(pub_result.data, result_register).get_counts()
                    elif hasattr(pub_result.data, "c"):
                        counts = pub_result.data.c.get_counts()
                    elif hasattr(pub_result.data, "meas"):
                        counts = pub_result.data.meas.get_counts()
                    else:
                        # Fallback - mark as failed
                        updates.append({"id": trial_id, "counts": None, "is_failed": True})
                        continue
                    
                    updates.append({"id": trial_id, "counts": counts})
                except Exception as e:
                    logger.warning(f"Error processing trial {trial_id}: {e}")
                    updates.append({"id": trial_id, "is_failed": True})
            
            # Perform bulk update
            if updates:
                with self.session() as session:
                    session.execute(
                        update(self.trial_class),
                        updates
                    )
                    session.commit()
                logger.info(f"Updated {len(updates)} trials for job {job_id}")
        except Exception as e:
            logger.error(f"Error updating trials for job {job_id}: {e}")

    async def update_all_pending_results(self, service, batch_size: int = 5, result_register: Optional[str] = None) -> None:
        """
        Update all pending job results asynchronously.

        Args:
            service: QiskitRuntimeService instance
            batch_size: Number of concurrent job fetches
            result_register: (Optional) "c", "meas", etc. to specify which register to extract counts from. Default: check "c", then "meas".
        """
        pending_jobs = self.get_pending_job_ids()

        if not pending_jobs:
            logger.info("No pending jobs to update")
            return

        logger.info(f"Updating {len(pending_jobs)} pending jobs")

        # Process jobs in batches to avoid overwhelming the API
        for i in range(0, len(pending_jobs), batch_size):
            batch = pending_jobs[i : i + batch_size]
            tasks = [self.update_job_results(job_id, service, result_register=result_register) for job_id in batch]

            batch_num = i // batch_size + 1
            total_batches = (len(pending_jobs) + batch_size - 1) // batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            await asyncio.gather(*tasks, return_exceptions=True)

    # Statistics and Maintenance
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        with self.session() as session:
            # Count problems
            total_problems = session.query(func.count(self.problem_class.id)).scalar()
            
            # Count trials
            total_trials = session.query(func.count(self.trial_class.id)).scalar()
            
            pending_trials = session.query(func.count(self.trial_class.id)).where(
                self.trial_class.counts == None,
                self.trial_class.job_id != None
            ).scalar()
            
            # Count failed trials (contains "-1" key)
            failed_trials = session.query(func.count(self.trial_class.id)).where(
                self.trial_class.counts.contains({"-1": 1})
            ).scalar() or 0
            
            # Compiler stats
            compile_stats_query = (
                select(
                    self.trial_class.compiler_name,
                    func.count(self.trial_class.id)
                )
                .group_by(self.trial_class.compiler_name)
            )
            compile_stats = dict(session.execute(compile_stats_query).all())
            
            avg_trials = total_trials / total_problems if total_problems > 0 else 0
            
            return {
                "problem_type": self.problem_type,
                "problem_instances": total_problems,
                "trials": {
                    "total": total_trials,
                    "pending": pending_trials or 0,
                    "failed": failed_trials,
                    "completed": total_trials - (pending_trials or 0) - failed_trials,
                    "by_compiler_name": compile_stats,
                    "avg_per_problem": round(avg_trials, 2),
                },
            }

    def calculate_trial_success_rate(self, trial: BaseTrial) -> float:
        """
        Calculate success rate for a trial, automatically loading problem instance.

        Args:
            trial: Trial to calculate success rate for

        Returns:
            Success rate between 0 and 1
        """
        try:
            trial.problem
        except DetachedInstanceError:
            trial = self.get_trial(trial.id)
        return trial.calculate_success_rate()

    def calculate_trial_expected_success_rate(self, trial: BaseTrial) -> float:
        """
        Calculate expected success rate for a trial, automatically loading problem instance.

        Args:
            trial: Trial to calculate expected success rate for

        Returns:
            Expected success rate between 0 and 1
        """
        try:
            trial.problem
        except DetachedInstanceError:
            trial = self.get_trial(trial.id)
        return trial.calculate_expected_success_rate()

    def get_trial_with_success_rates(
        self, trial_id: int
    ) -> Tuple[BaseTrial, float, float]:
        """
        Get trial with computed success rates.

        Args:
            trial_id: ID of trial to retrieve

        Returns:
            Tuple of (trial, actual_success_rate, expected_success_rate)
        """
        trial = self.get_trial(trial_id)
        actual_rate = self.calculate_trial_success_rate(trial)
        expected_rate = self.calculate_trial_expected_success_rate(trial)
        return trial, actual_rate, expected_rate

    def recompute_simulations(self, instance_ids: Optional[List[int]] = None) -> None:
        """
        Recompute simulation results for trials.

        Args:
            instance_ids: Limit to specific problem instances (None for all)
        """
        # Find trials to recompute
        if instance_ids:
            trials = []
            for instance_id in instance_ids:
                trials.extend(self.find_trials(instance_id=instance_id))
        else:
            trials = self.find_trials()

        logger.info(f"Recomputing simulations for {len(trials)} trials")

        for trial in trials:
            # Load problem instance and recompute simulation
            problem = self.get_problem_instance(trial.instance_id)

            # This would need to be implemented by specific trial types
            if hasattr(trial, "recompute_simulation"):
                trial.recompute_simulation(problem)
                self.save_trial(trial)

def hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance between two binary strings."""
    if len(s1) != len(s2):
        raise ValueError("Strings must have equal length")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))
