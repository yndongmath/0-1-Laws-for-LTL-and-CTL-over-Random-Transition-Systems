"""The fixed formulas of Table 2 (a_1..a_5 are indices 0..4)."""


def a(i):
    return ('ap', i - 1)


def neg(f):
    return ('not', f)


LTL_FIXED = [
    ('X', a(1)),                                                     # psi1
    ('G', a(1)),                                                     # psi2
    ('U', a(1), a(2)),                                               # psi3
    ('F', ('and', a(1), ('F', a(2)))),                               # psi4
    ('G', ('imp', a(1), ('F', a(2)))),                               # psi5
    ('F', ('U', ('and', a(1), a(2)), a(3))),                         # psi6
    ('G', ('or', neg(a(1)), ('X', a(2)))),                           # psi7
    ('F', ('and', a(1), ('and', ('X', ('or', a(2), a(3))), ('F', a(4))))),  # psi8
]

_phi7_inner = ('and', a(2), ('EX', ('EU', a(3), a(4))))
_phi9_lhs = ('EF', ('and', a(1), ('EX', a(2))))
_phi9_rhs = ('EG', ('or', neg(a(3)), ('AU', a(4), ('AF', a(5)))))
_phi10_left = ('imp', a(1), ('AF', ('and', a(2), ('EX', ('or', a(3), ('EU', neg(a(3)), a(4)))))))
_phi10_right = ('EG', ('and', neg(a(2)), ('EU', a(1), ('or', a(4), ('AF', a(5))))))

CTL_FIXED = [
    ('AF', a(1)),                                            # phi1
    ('EF', ('EX', a(3))),                                    # phi2
    ('EF', ('and', a(1), ('EF', a(2)))),                     # phi3
    ('EG', ('and', a(1), ('or', a(2), a(3)))),               # phi4
    ('AG', ('imp', a(1), ('AF', a(2)))),                     # phi5
    ('AU', ('imp', a(1), a(2)), a(3)),                       # phi6
    ('AG', ('imp', a(1), ('AF', _phi7_inner))),              # phi7
    ('AG', ('imp', a(1), ('AF', ('EX', neg(a(2)))))),        # phi8
    ('AG', ('imp', _phi9_lhs, _phi9_rhs)),                   # phi9
    ('AG', ('or', _phi10_left, _phi10_right)),               # phi10
]
