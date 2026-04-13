# benchmarklib
A utility library for benchmarking quantum circuits. benchmarklib makes it easy to execute large quantities of circuits, store results in an SQLite database, and gain insights from experiments. Built with compiler benchmarking in mind, benchmarklib supports configurable compile+transpile pipelines.

## Installation
Local installation is recommended
```bash
git clone https://github.com/OpenPyQ/benchmarklib.git
pip install ./benchmarklib
```

## Usage

### Database Setup
There are two important subclasses you need to create to use benchmarklib:

**Problem**: specifies a kind of circuit, problem instance, or algorithm configuration that needs to be benchmarked

**Trial**: specifies a specific execution instance for a problem

For instance, if your benchmark involves comparing an algorithm's performance across backends, you would use a single problem
to represent the algorithm instance and as many trials as needed for each run on each backend.

Each class corresponds to a database table via `sqlalchemy`.

Here is a simple example:
```py
from sqlalchemy import Column
from sqlalchemy.orm import Mapped
from benchmarklib import BaseProblem, BaseTrial, BenchmarkDatabase

class Problem(BaseProblem):
    __tablename__ = "problems"
    TrialClass = "Trial"

    name: Mapped[str] = Column(String(128), nullable=False, index=True)

    # other attributes if desired...
        

class Trial(BaseTrial):
    __tablename__ = "trials"
    ProblemClass = Problem

    # other attributes if desired...
```

### Database usage
Load an sqlite database (creating the file if it does not yet exist):
```py
from benchmarklib import BenchmarkDatabase

# use the Problem and Trial classes you created earlier
db = BenchmarkDatabase("my_database.db", Problem, Trial)
```

You can use access the database in the same way as any sqlalchemy ORM database with the `session` method:
```py
from sqlalchemy import select

with db.session() as session:
    query = select(Problem)
            .where(Problem.name == "test")
            .limit(10)
    results = session.execute(query).scalars().all()
```

Or use the `query` method which acts as a wrapper around the above session execution. This is useful for common database lookups,
but for repeated queries or use cases that require something other than `.scalars().all()` it is recommended to use the session directly.
```py
from sqlalchemy import select

results = db.query(
    select(Problem)
    .where(Problem.name == "test")
    .limit(10)
)
```

### Circuit Batching
IBM supports job batching and multi-circuit jobs. To maximize throughput when running many circuits, benchmarklib provides a utility to automatically collect circuits together to submit in jobs.
`BatchQueue` provides the following:
* `enqueue`: add a trial and its circuit to the queue to be submitted (circuits are submitted once the batch size is reached or when the BatchQueue context manager is exited)
* Batch size (default 100) circuits are submitted per job for efficiency
* Binary backoff: if the batch is too large to submit as a single job, this repeatedly tries with half the number of circuits until finding a suitable batch size
* Autosave: once a circuit is submitted in a job, its trial is saved to the database with updated `job_id` and `job_pub_idx` as identifiers for fetching results.

```py
from benchmarklib import BatchQueue

# get backend
service = QiskitRuntimeService()
backend = service.least_busy()

with BatchQueue(db, backend=backend, shots=1024) as q:
    for trial in trials: # iterate over generator or iterator of Trial instances
        q.enqueue(trial, trial.circuit, run_simulation=False)
```

Later, once jobs have finished, update the local database with results:
```py
await db.update_all_pending_trials(service=service)
```

Or, if you only want to update results from a particular job id:
```py
await db.update_job_results(job_id, service)
```

### Manage Processing Resources
Sometimes, benchmarks involve operations that can consume large amounts of RAM or spin in an endless loop. benchmarklib provides a utility to run these kinds of tasks as separate processes and constrain their resource usage. This way, a bad instance can have its process killed without affecting the rest of the benchmark. 
```py
def my_hungry_function(var_a, var_b):
    pass

ex = run_with_resource_limits(
    my_hungry_function,
    kwargs={
        "var_a": ,
        "var_b": 
    },
    memory_limit_mb=2024,
    timeout_seconds=300
)

if ex.success:
    return_val = ex.result
else:
    print(f"Execution killed. Reason: {ex.error_message}")
```

No more failed overnight benchmarks due to a bad problem instance!