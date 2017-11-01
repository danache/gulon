import numpy as np
import mxnet as mx
# def getTransform(center, scale, rot ,res):
#     h = 200 * scale
#     t = np.eye(3)
#     #scale
#     t[0][0] = res *1./ h
#     t[1][1] = res * 1. / h
#     #translation
#     t[0][2] = res * (-center[0] * 1. / h + .5)
#     t[1][2] = res * (-center[1] * 1. / h + .5)
#     #roate
#     if rot != 0:
#         rot = -rot
#         r = np.eye(3)
#         ang = rot * np.pi / 180
#         s = np.sin(ang)
#         c = np.cos(ang)
#         r[0][0] = c
#         r[0][1] = -s
#         r[1][0] = s
#         r[1][1] = c
#
#         t_ = np.eye(3)
#
#         t_[0][2] = -res / 2
#         t_[1][2] = -res / 2
#
#         t_inv = np.eye(3)
#         t_inv[0][2] = res / 2
#         t_inv[1][2] = res / 2
#         t = t_inv * r * t_ * t
#     return t
#
# def transform(pt, center, scale, rot, res, invert):
#     pt_ = np.ones(3)
#     pt_[0], pt_[1] = pt[0] - 1 ,pt[1] - 1
#     t = getTransform(center,scale,rot, res)
#     if invert:
#         t = np.reverse(t)
#
