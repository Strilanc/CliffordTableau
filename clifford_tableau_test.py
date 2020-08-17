import itertools
import random
from typing import Any, Tuple, List

import cirq
import numpy as np
import pytest

from clifford_tableau import CliffordTableau, _known_gate_conversions


@pytest.mark.parametrize("gate", _known_gate_conversions())
def test_known_gate_conversions(gate: cirq.Gate):
    assert_mapped_correctly(gate)


def assert_mapped_correctly(obj: Any) -> bool:
    tab = CliffordTableau(obj)
    known_unitary = cirq.unitary(obj)
    num_qubits = known_unitary.shape[0].bit_length() - 1
    qs = cirq.LineQubit.range(num_qubits)
    gate = cirq.MatrixGate(known_unitary)
    for q in qs:
        for pre in [cirq.X(q), cirq.Z(q)]:
            post = tab(pre)
            if not np.allclose(
                    cirq.unitary(cirq.Circuit(pre, gate(*qs), post)),
                    cirq.unitary(gate),
                    atol=1e-8):
                assert False, (
                    f"Not equivalent.\n"
                    f"Bad pre: {pre}\n"
                    f"Bad post: {post}\n"
                    f"\n"
                    f"Known reference:\n"
                    f"{obj}\n"
                    f"\n"
                    f"Tableau:\n"
                    f"{tab}"
                )

    return all(
        np.allclose(
            cirq.unitary(cirq.Circuit(pre, gate(*qs), tab(pre))),
            cirq.unitary(gate),
            atol=1e-8)
        for q in qs
        for pre in [cirq.X(q), cirq.Z(q)]
    )


def test_all_single_qubit_inverses():
    q = cirq.LineQubit(0)
    for a, b, c in itertools.product([cirq.I, cirq.S, cirq.Z, cirq.S**-1], repeat=3):
        assert_mapped_correctly(cirq.Circuit(a(q), cirq.H(q), b(q), cirq.H(q), c(q)))


def test_identity():
    q = cirq.NamedQubit('q')

    i = CliffordTableau(cirq.I(q))

    assert i(cirq.X(q)) == cirq.X(q)
    assert i(cirq.Y(q)) == cirq.Y(q)
    assert i(cirq.Z(q)) == cirq.Z(q)
    assert i(-cirq.X(q)) == -cirq.X(q)

    assert i.inverse() == i

    assert str(i).strip() == """
      |
------+-
 sign |
""".strip()


def test_paulis():
    q = cirq.NamedQubit('q')

    x = CliffordTableau(cirq.X(q))
    y = CliffordTableau(cirq.Y(q))
    z = CliffordTableau(cirq.Z(q))

    assert x(cirq.X(q)) == cirq.X(q)
    assert y(cirq.X(q)) == -cirq.X(q)
    assert z(cirq.X(q)) == -cirq.X(q)

    assert x(cirq.Y(q)) == -cirq.Y(q)
    assert y(cirq.Y(q)) == cirq.Y(q)
    assert z(cirq.Y(q)) == -cirq.Y(q)

    assert x(cirq.Z(q)) == -cirq.Z(q)
    assert y(cirq.Z(q)) == -cirq.Z(q)
    assert z(cirq.Z(q)) == cirq.Z(q)

    assert x.inverse() == x
    assert y.inverse() == y
    assert z.inverse() == z

    assert str(x).strip() == """
      | q
------+-xz-
 q    | XZ
 sign | +-
    """.strip()

    assert str(y).strip() == """
      | q
------+-xz-
 q    | XZ
 sign | --
    """.strip()

    assert str(z).strip() == """
      | q
------+-xz-
 q    | XZ
 sign | -+
    """.strip()


def test_hadamard():
    q = cirq.NamedQubit('q')

    h = CliffordTableau(cirq.H(q))

    assert h(cirq.X(cirq.NamedQubit('other'))) == cirq.X(cirq.NamedQubit('other'))
    assert h(cirq.Y(cirq.NamedQubit('other'))) == cirq.Y(cirq.NamedQubit('other'))
    assert h(cirq.Z(cirq.NamedQubit('other'))) == cirq.Z(cirq.NamedQubit('other'))

    assert h(cirq.X(q) * cirq.X(cirq.NamedQubit('other'))) == cirq.Z(q) * cirq.X(cirq.NamedQubit('other'))
    assert h(-cirq.X(q)) == -cirq.Z(q)

    assert h(cirq.X(q)) == cirq.Z(q)
    assert h(cirq.Y(q)) == -cirq.Y(q)
    assert h(cirq.Z(q)) == cirq.X(q)

    assert h.then(h) == CliffordTableau(cirq.I)
    assert h.inverse() == h

    assert str(h).strip() == """
      | q
------+-xz-
 q    | ZX
 sign | ++
    """.strip()


def test_cz():
    a, b = cirq.NamedQubit('a'), cirq.NamedQubit('b')

    cz = CliffordTableau(cirq.CZ(a, b))

    assert cz(cirq.X(a)) == cirq.X(a) * cirq.Z(b)
    assert cz(cirq.Z(a)) == cirq.Z(a)
    assert cz(cirq.X(a) * cirq.X(b)) == cirq.Y(a) * cirq.Y(b)
    assert cz(cirq.X(a) * cirq.Y(b)) == -cirq.Y(a) * cirq.X(b)

    assert cz.then(cz) == CliffordTableau(cirq.I)
    assert cz.inverse() == cz

    assert str(cz).strip() == """
      | a  b
------+-xz-xz-
 a    | XZ Z_
 b    | Z_ XZ
 sign | ++ ++
    """.strip()


def test_cnot():
    a, b = cirq.NamedQubit('a'), cirq.NamedQubit('b')

    cnot = CliffordTableau(cirq.CNOT(a, b))

    assert cnot(cirq.X(a)) == cirq.X(a) * cirq.X(b)
    assert cnot(cirq.Z(a)) == cirq.Z(a)
    assert cnot.inverse().then(cnot) == CliffordTableau(cirq.I)
    assert cnot.inverse() == cnot
    assert cnot.then(cnot) == CliffordTableau(cirq.I)

    ha = CliffordTableau(cirq.H(a))
    hb = CliffordTableau(cirq.H(b))
    assert hb.then(cnot).then(hb) == CliffordTableau(cirq.CZ(a, b))
    assert ha.then(hb).then(cnot).then(ha).then(hb) == CliffordTableau(cirq.CNOT(b, a))

    assert str(cnot).strip() == """
      | a  b
------+-xz-xz-
 a    | XZ _Z
 b    | X_ XZ
 sign | ++ ++
    """.strip()


def test_phase():
    q = cirq.NamedQubit('q')

    s = CliffordTableau(cirq.S(q))
    s_dag = CliffordTableau(cirq.S(q) ** -1)

    assert s(cirq.X(q)) == cirq.Y(q)
    assert s(cirq.Y(q)) == -cirq.X(q)
    assert s(cirq.Z(q)) == cirq.Z(q)

    assert s_dag(cirq.X(q)) == -cirq.Y(q)
    assert s_dag(cirq.Y(q)) == cirq.X(q)
    assert s_dag(cirq.Z(q)) == cirq.Z(q)

    assert s.then(s) == CliffordTableau(cirq.Z(q))
    assert s.then(s).then(s) == s_dag
    assert s.then(s_dag) == CliffordTableau(cirq.I)

    assert s.inverse() == s_dag
    assert s_dag.inverse() == s

    assert str(s).strip() == """
      | q
------+-xz-
 q    | YZ
 sign | ++
       """.strip()

    assert str(s_dag).strip() == """
      | q
------+-xz-
 q    | YZ
 sign | -+
    """.strip()


def test_swap():
    a, b, c = cirq.LineQubit.range(3)

    ab = CliffordTableau(cirq.SWAP(a, b))
    ac = CliffordTableau(cirq.SWAP(a, c))
    bc = CliffordTableau(cirq.SWAP(b, c))

    assert ab.then(bc).then(ab) == ac
    backward = ab.then(bc)
    forward = bc.then(ab)
    assert forward.then(forward) == backward
    assert backward.then(backward) == forward
    assert forward.then(forward).then(forward) == CliffordTableau(cirq.I)
    assert forward.inverse() == backward
    assert backward.inverse() == forward

    assert str(forward).strip() == """
      | 0  1  2
------+-xz-xz-xz-
 0    | __ __ XZ
 1    | XZ __ __
 2    | __ XZ __
 sign | ++ ++ ++
       """.strip()

    assert str(backward).strip() == """
      | 0  1  2
------+-xz-xz-xz-
 0    | __ XZ __
 1    | __ __ XZ
 2    | XZ __ __
 sign | ++ ++ ++
       """.strip()


def test_cnot_chain():
    a, b, c = cirq.LineQubit.range(3)

    ab = CliffordTableau(cirq.CNOT(a, b))
    bc = CliffordTableau(cirq.CNOT(b, c))
    ac = CliffordTableau(cirq.CNOT(a, c))

    assert bc.then(ab).then(bc).then(ab) == ac
    assert ab.then(bc).then(ab) == ac.then(bc)
    assert ab.then(bc).inverse() == bc.then(ab)
    assert bc.then(ab).then(bc).inverse() == bc.then(ab).then(bc)


def sample_circuit() -> cirq.Circuit:
    qs = cirq.LineQubit.range(4)
    circuit = cirq.Circuit()
    for n in range(50):
        circuit.append(cirq.H(random.choice(qs)))
        circuit.append(cirq.S(random.choice(qs)))
        circuit.append(cirq.CNOT(*random.sample(qs, 2)))
    return circuit


def test_fuzz():
    circuit = sample_circuit()
    assert CliffordTableau(circuit).inverse() == CliffordTableau(circuit ** -1)
    print(CliffordTableau(circuit))
    print(CliffordTableau(circuit).inverse())
    assert_mapped_correctly(circuit)


def _bit_table(tableau: CliffordTableau) -> Tuple[List[cirq.Qid], np.ndarray]:
    involved = sorted(tableau._dat.x_out.keys() | tableau._dat.z_out.keys())
    q2i = {q: i for i, q in enumerate(involved)}
    q2x = {q: 2*i for q, i in q2i.items()}
    q2z = {q: 2*i + 1 for q, i in q2i.items()}
    result = np.zeros(shape=2 * (2 * len(involved),), dtype=np.int8)
    for q_in in involved:
        for d, q2j in [(tableau(cirq.X(q_in)), q2x), (tableau(cirq.Z(q_in)), q2z)]:
            for q_out, p in d.items():
                if p != cirq.Z:
                    result[q2j[q_in], q2z[q_out]] = 1
                if p != cirq.X:
                    result[q2j[q_in], q2x[q_out]] = 1
    return involved, result


def test_inverse_bit_table_is_transpose():
    t = CliffordTableau(sample_circuit())
    _, b = _bit_table(t)
    _, b_inv = _bit_table(t.inverse())
    np.testing.assert_equal(b.transpose(), b_inv)
