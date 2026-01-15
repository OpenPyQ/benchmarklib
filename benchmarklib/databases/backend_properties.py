import os

from benchmarklib.core.database import BackendPropertyManager

def get_backend_property_manager(db_path: str = None) -> BackendPropertyManager:
    """Get the BackendPropertyManager instance for the given database path.

    If no path is provided, it defaults to 'backend_properties.db' in the current directory.
    """
    if db_path is None:
        db_path = os.path.join(os.path.dirname(__file__), "backend_properties.db")
    return BackendPropertyManager(db_path)

if __name__ == "__main__":
    import datetime
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    backend = service.backend(name="ibm_rensselaer")
    backend_properties_db = get_backend_property_manager()
    backend_properties_db.load_missing_dates(backend, datetime.date(2025, 1, 1), datetime.date.today())
