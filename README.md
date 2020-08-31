Represents Clifford operations using a table where each column says how that Clifford operation conjugates a generator of the Pauli group.

Example usage:

```python
import cirq
from clifford_tableau import CliffordTableau

a, b = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(b), cirq.CNOT(a, b), cirq.H(b))
tableau = CliffordTableau(circuit)
print(tableau)
#       | 0  1
# ------+-xz-xz-
#  0    | XZ Z_
#  1    | Z_ XZ
#  sign | ++ ++

print(tableau(cirq.X(a)))
# X(0)*Z(1)

print(tableau(cirq.X(a) * cirq.Y(b)))
# -Y(0)*X(1)

assert tableau == CliffordTableau(cirq.CZ(a, b))
s = CliffordTableau(cirq.S(a))
assert s.inverse() == CliffordTableau(cirq.S(a)**-1) != s
assert s.then(s) == CliffordTableau(cirq.Z(a))
```
