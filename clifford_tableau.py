import functools
import itertools
from typing import Dict, Tuple, Callable, Union

import cirq


class CliffordTableau:
    """Transforms Pauli products into Pauli products.

    Examples:
        >>> import cirq
        >>> a, b = cirq.LineQubit.range(2)
        >>> circuit = cirq.Circuit(cirq.H(b), cirq.CNOT(a, b), cirq.H(b))
        >>> tableau = CliffordTableau(circuit)
        >>> print(tableau)
              | 0  1
        ------+-xz-xz-
         0    | XZ Z_
         1    | Z_ XZ
         sign | ++ ++
        >>> print(tableau(cirq.X(a)))
        X(0)*Z(1)
        >>> print(tableau(cirq.X(a) * cirq.Y(b)))
        -Y(0)*X(1)
        >>> assert tableau == CliffordTableau(cirq.CZ(a, b))
        >>> s = CliffordTableau(cirq.S(a))
        >>> assert s.inverse() == CliffordTableau(cirq.S(a)**-1) != s
        >>> assert s.then(s) == CliffordTableau(cirq.Z(a))
    """

    def __init__(self, content: Union[cirq.Gate, cirq.Operation, cirq.Circuit, '_TabInternalData']):
        if isinstance(content, cirq.Gate):
            content = _known_gate_conversions()[content]
        elif isinstance(content, cirq.Operation):
            qs = content.qubits
            content = _known_gate_conversions()[content.gate].transform_qubits(lambda q: qs[q.x])
        elif isinstance(content, cirq.Circuit):
            tab = CliffordTableau(cirq.I)
            for op in content.all_operations():
                tab = tab.then(CliffordTableau(op))
            content = tab._dat

        if not isinstance(content, _TabInternalData):
            raise ValueError(f"not isinstance({content!r}, "
                             f"(cirq.Gate, cirq.Circuit, cirq.Operation, _TabInternalData))")

        self._dat = content

    def __call__(self, observable: cirq.PauliString) -> cirq.PauliString:
        """Returns the Pauli product that the given Pauli product is mapped to by this tableau."""
        result = _ID * observable.coefficient
        for qubit, pauli in observable.items():
            if pauli == cirq.X:
                result *= self._dat.x_image(qubit)
            elif pauli == cirq.Z:
                result *= self._dat.z_image(qubit)
            else:  # pauli == cirq.Y
                result *= self._dat.y_image(qubit)
        return result

    def then(self, other: 'CliffordTableau') -> 'CliffordTableau':
        """Returns a tableau equal to the composition of the receiving tableau and the given tableau."""
        involved = self._dat.x_out.keys() | self._dat.z_out.keys() | other._dat.x_out.keys() | other._dat.z_out.keys()

        x_out = {}
        z_out = {}
        for q in involved:
            x_out[q] = other(self(cirq.X(q)))
            z_out[q] = other(self(cirq.Z(q)))

        return CliffordTableau(_TabInternalData(x_out, z_out))

    def inverse(self) -> 'CliffordTableau':
        """Returns a tableau equal to the inverse of the receiving tableau."""
        x_out = {}
        z_out = {}
        involved = self._dat.x_out.keys() | self._dat.z_out.keys()

        for in_qubit in involved:
            x_img = self._dat.x_image(in_qubit)
            z_img = self._dat.z_image(in_qubit)
            for out_qubit in x_img.keys() | z_img.keys():
                inv_x, inv_z = _inverse_flow(x_img.get(out_qubit, cirq.I), z_img.get(out_qubit, cirq.I))
                x_out.setdefault(out_qubit, _ID)
                z_out.setdefault(out_qubit, _ID)
                x_out[out_qubit] *= inv_x(in_qubit)
                z_out[out_qubit] *= inv_z(in_qubit)

        for d in x_out, z_out:
            for q, ps in list(d.items()):
                d[q] *= self(ps).coefficient

        return CliffordTableau(_TabInternalData(x_out, z_out))

    def __eq__(self, other: 'CliffordTableau') -> bool:
        if not isinstance(other, CliffordTableau):
            return NotImplemented
        return self._dat.x_out == other._dat.x_out and self._dat.z_out == other._dat.z_out

    def __ne__(self, other: 'CliffordTableau') -> bool:
        return not self == other

    def transform_qubits(self, func: Callable[[cirq.Qid], cirq.Qid]):
        return CliffordTableau(self._dat.transform_qubits(func))

    def __str__(self):
        diagram = cirq.TextDiagramDrawer()

        involved = set()
        for d in self._dat.x_out, self._dat.z_out:
            for q, p in d.items():
                involved.add(q)
                involved |= p.keys()
        q2i = {q: i + 3 for i, q in enumerate(sorted(involved))}
        sign = max(q2i.values(), default=2) + 1

        for q in involved:
            diagram.write(q2i[q], 1, str(q))
            diagram.write(q2i[q], 2, 'xz')
            diagram.write(1, q2i[q], str(q))
        diagram.write(1, sign, "sign")

        sign_map: Dict[complex, str] = {+1: '+', -1: '-'}
        pauli_map = {cirq.X: 'X', cirq.Y: 'Y', cirq.Z: 'Z', cirq.I: '_'}
        for q in involved:
            x = self(cirq.X(q))
            z = self(cirq.Z(q))
            for q2 in involved:
                diagram.write(q2i[q], q2i[q2], pauli_map[x.get(q2, cirq.I)] + pauli_map[z.get(q2, cirq.I)])
            diagram.write(q2i[q], sign, sign_map[x.coefficient] + sign_map[z.coefficient])

        diagram.horizontal_line(y=2, x1=0, x2=sign)
        diagram.vertical_line(x=2, y1=0, y2=sign + 1)

        result = diagram.render(horizontal_spacing=1, vertical_spacing=0, use_unicode_characters=False)
        return '\n'.join(line[1:] for line in result.split('\n')[1:])

    def __repr__(self):
        return f'Tab(_TabInternalData(\n    x_out={self._dat.x_out!r},\n    z_out={self._dat.z_out!r},\n)'


class _TabInternalData:
    def __init__(self, x_out: Dict[cirq.Qid, cirq.PauliString], z_out: Dict[cirq.Qid, cirq.PauliString]):
        self.x_out = {q: p for q, p in x_out.items() if p != cirq.X(q)}
        self.z_out = {q: p for q, p in z_out.items() if p != cirq.Z(q)}
        self.check_invariants()

    def transform_qubits(self, func: Callable[[cirq.Qid], cirq.Qid]):
        return _TabInternalData(
            x_out={func(q): p.transform_qubits(func) for q, p in self.x_out.items()},
            z_out={func(q): p.transform_qubits(func) for q, p in self.z_out.items()},
        )

    def x_image(self, qubit) -> cirq.PauliString:
        result = self.x_out.get(qubit)
        if result is not None:
            return result
        return cirq.X(qubit)

    def y_image(self, qubit) -> cirq.PauliString:
        # Y = i*X*Z
        return 1j * self.x_image(qubit) * self.z_image(qubit)

    def z_image(self, qubit) -> cirq.PauliString:
        result = self.z_out.get(qubit)
        if result is not None:
            return result
        return cirq.Z(qubit)

    def check_invariants(self):
        involved = set()
        for d in self.x_out, self.z_out:
            for q, p in d.items():
                involved.add(q)
                involved |= p.keys()

        for qubit in involved:
            x1 = self.x_image(qubit)
            z1 = self.z_image(qubit)
            assert x1.coefficient in [-1, +1]
            assert z1.coefficient in [-1, +1]
            assert not cirq.commutes(x1, z1)

        for q1, q2 in itertools.combinations(involved, 2):
            x1 = self.x_image(q1)
            x2 = self.x_image(q2)
            z1 = self.z_image(q1)
            z2 = self.z_image(q2)
            assert cirq.commutes(x1, x2)
            assert cirq.commutes(x1, z2)
            assert cirq.commutes(z1, x2)
            assert cirq.commutes(z1, z2)


@functools.lru_cache(maxsize=1)
def _known_gate_conversions() -> Dict[cirq.Gate, _TabInternalData]:
    a, b = cirq.LineQubit.range(2)
    return {
        cirq.I: _TabInternalData(x_out={}, z_out={}),
        cirq.H: _TabInternalData(
            x_out={a: cirq.Z(a)},
            z_out={a: cirq.X(a)},
        ),
        cirq.X**0.5: _TabInternalData(
            x_out={},
            z_out={a: -cirq.Y(a)},
        ),
        cirq.X**-0.5: _TabInternalData(
            x_out={},
            z_out={a: cirq.Y(a)},
        ),
        cirq.Y**0.5: _TabInternalData(
            x_out={a: -cirq.Z(a)},
            z_out={a: cirq.X(a)},
        ),
        cirq.Y**-0.5: _TabInternalData(
            x_out={a: cirq.Z(a)},
            z_out={a: -cirq.X(a)},
        ),
        cirq.S: _TabInternalData(
            x_out={a: cirq.Y(a)},
            z_out={},
        ),
        cirq.S**-1: _TabInternalData(
            x_out={a: -cirq.Y(a)},
            z_out={},
        ),
        cirq.CZ: _TabInternalData(
            x_out={a: cirq.X(a) * cirq.Z(b), b: cirq.Z(a) * cirq.X(b)},
            z_out={},
        ),
        cirq.CNOT: _TabInternalData(
            x_out={a: cirq.X(a) * cirq.X(b)},
            z_out={b: cirq.Z(a) * cirq.Z(b)},
        ),
        cirq.X: _TabInternalData(
            x_out={},
            z_out={a: -cirq.Z(a)},
        ),
        cirq.Y: _TabInternalData(
            x_out={a: -cirq.X(a)},
            z_out={a: -cirq.Z(a)},
        ),
        cirq.Z: _TabInternalData(
            x_out={a: -cirq.X(a)},
            z_out={},
        ),
        cirq.SWAP: _TabInternalData(
            x_out={a: cirq.X(b), b: cirq.X(a)},
            z_out={a: cirq.Z(b), b: cirq.Z(a)},
        ),
        cirq.ISWAP: _TabInternalData(
            x_out={a: cirq.Z(a) * cirq.Y(b), b: cirq.Y(a) * cirq.Z(b)},
            z_out={a: cirq.Z(b), b: cirq.Z(a)},
        ),
        cirq.ISWAP**-1: _TabInternalData(
            x_out={a: -cirq.Z(a) * cirq.Y(b), b: -cirq.Y(a) * cirq.Z(b)},
            z_out={a: cirq.Z(b), b: cirq.Z(a)},
        ),
    }


@functools.lru_cache(maxsize=16)
def _inverse_flow(image_x: cirq.Pauli, image_z: cirq.Pauli) -> Tuple[cirq.Pauli, cirq.Pauli]:
    c_xx = cirq.commutes(image_x, cirq.X)
    c_xz = cirq.commutes(image_x, cirq.Z)
    c_zx = cirq.commutes(image_z, cirq.X)
    c_zz = cirq.commutes(image_z, cirq.Z)

    matches_x = [
        px
        for px in [cirq.I, cirq.X, cirq.Y, cirq.Z]
        if c_xx == cirq.commutes(px, cirq.X)
        if c_zx == cirq.commutes(px, cirq.Z)
    ]

    matches_z = [
        pz
        for pz in [cirq.I, cirq.X, cirq.Y, cirq.Z]
        if c_xz == cirq.commutes(pz, cirq.X)
        if c_zz == cirq.commutes(pz, cirq.Z)
    ]

    assert len(matches_x) == len(matches_z) == 1

    return matches_x[0], matches_z[0]


_ID = cirq.PauliString()
