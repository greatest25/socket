import math

def cal_triangle_S(p1, p2, p3):
    S = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
    if math.isclose(S, 0.0, abs_tol=1e-9):
        return 0.0
    else:
        return S