import datetime
import numpy as np

from qiskit.providers import Backend

from benchmarklib.databases.backend_properties import get_backend_property_manager

backend_db = get_backend_property_manager()
cache = {}

def get_mean_gate_errors(backend: Backend, at: datetime.date):
    if at in cache:
        return cache[at]
    props = backend_db.latest(backend, at)
    cache[at] = props.get_average_gate_errors()
    return cache[at]

med_cache = {}
def get_median_gate_errors(backend: Backend, at: datetime.date):
    if at in med_cache:
        return med_cache[at]
    props = backend_db.latest(backend, at)
    med_cache[at] = {g : np.median(v) for g, v in props.get_gate_errors().items()}
    return med_cache[at]

def compute_accumulated_gate_error1(circuit, gate_errors):
    assert circuit is not None
    expected_success_rate = 1.0
    for inst in circuit.data:
        expected_success_rate *= (1 - gate_errors.get(inst[0].name, 0.0))
    return expected_success_rate

def compute_analytic_success_rate_estimate1(backend: Backend, circuit, created_at=None):
    created_at_date = created_at.date() if created_at is not None else datetime.date.today()
    analytic_estimate1 = compute_accumulated_gate_error1(circuit, gate_errors = get_median_gate_errors(backend, created_at_date))
    analytic_estimate2 = compute_accumulated_gate_error1(circuit, gate_errors = get_mean_gate_errors(backend, created_at_date))
    return (analytic_estimate1 + analytic_estimate2) / 2