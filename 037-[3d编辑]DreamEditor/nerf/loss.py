# MIT License
#
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import numpy as np



class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False, size_factor=1.0):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        self.size_factor = size_factor
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            if laplacian[i, i] != 0: laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x * self.size_factor

        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)

        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).sum(dims)

        if self.average:
            return x.sum() / batch_size
        else:
            return x


class EdgeLoss(nn.Module):
    def __init__(self, faces, average=False, size_factor=1.0):
        super(EdgeLoss, self).__init__()
        edge_set = set()

        for tri in faces:
            tri = tri.numpy()

            edge_set.add((tri[0], tri[1]))
            edge_set.add((tri[1], tri[2]))
            edge_set.add((tri[0], tri[2]))
        self.edges = torch.tensor(np.array(list(edge_set))).long().cuda()
        self.size_factor = size_factor

    def forward(self, x):
        x = x * self.size_factor
        p1 = x[self.edges[:, 0]]
        p2 = x[self.edges[:, 1]]

        distance = nn.functional.pairwise_distance(p1, p2, p=2)

        return torch.std(distance)


class NormLoss(nn.Module):
    def __init__(self, norm):
        super(NormLoss, self).__init__()
        self.norm = norm.cuda()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        cos_theta = 1 - self.cos(x, self.norm).abs()
        loss = torch.mean(cos_theta)

        return loss


class FlattenLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(FlattenLoss, self).__init__()
        self.nf = faces.size(0)
        self.average = average

        faces = faces.detach().cpu().numpy()

        vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))
        vert_face = {}
        for k, v in enumerate(faces):
            for vx in v:
                if vx not in vert_face.keys():
                    vert_face[vx] = [k]
                else:
                    vert_face[vx].append(k)

        v0s = np.array([v[0] for v in vertices], 'int32')
        v1s = np.array([v[1] for v in vertices], 'int32')
        v2s = []
        v3s = []

        idx = 0
        nosin_list = []
        for v0, v1 in zip(v0s, v1s):
            count = 0
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) > 2:
                continue
            # for face in faces:
            if len(sorted(list(set(vert_face[v0]) & set(vert_face[v1])))) == 2:
                nosin_list.append(idx)

            for faceid in sorted(list(set(vert_face[v0]) & set(vert_face[v1]))):
                face = faces[faceid]
                if v0 in face and v1 in face:
                    v = np.copy(face)
                    v = v[v != v0]
                    v = v[v != v1]
                    if count == 0:
                        v2s.append(int(v[0]))
                        count += 1
                    else:
                        v3s.append(int(v[0]))
            idx += 1
        v2s = np.array(v2s, 'int32')
        v3s = np.array(v3s, 'int32')

        v0s = v0s[nosin_list]
        v1s = v1s[nosin_list]
        v2s = v2s[nosin_list]

        self.register_buffer('v0s', torch.from_numpy(v0s).long())
        self.register_buffer('v1s', torch.from_numpy(v1s).long())
        self.register_buffer('v2s', torch.from_numpy(v2s).long())
        self.register_buffer('v3s', torch.from_numpy(v3s).long())

    def forward(self, vertices, eps=1e-6):
        # make v0s, v1s, v2s, v3s

        vertices = vertices.unsqueeze(0)
        batch_size = vertices.size(0)

        v0s = vertices[:, self.v0s, :]
        v1s = vertices[:, self.v1s, :]
        v2s = vertices[:, self.v2s, :]
        v3s = vertices[:, self.v3s, :]

        a1 = v1s - v0s
        b1 = v2s - v0s
        a1l2 = a1.pow(2).sum(-1)
        b1l2 = b1.pow(2).sum(-1)
        a1l1 = (a1l2 + eps).sqrt()
        b1l1 = (b1l2 + eps).sqrt()
        ab1 = (a1 * b1).sum(-1)
        cos1 = ab1 / (a1l1 * b1l1 + eps)
        sin1 = (1 - cos1.pow(2) + eps).sqrt()
        c1 = a1 * (ab1 / (a1l2 + eps))[:, :, None]
        cb1 = b1 - c1
        cb1l1 = b1l1 * sin1

        a2 = v1s - v0s
        b2 = v3s - v0s
        a2l2 = a2.pow(2).sum(-1)
        b2l2 = b2.pow(2).sum(-1)
        a2l1 = (a2l2 + eps).sqrt()
        b2l1 = (b2l2 + eps).sqrt()
        ab2 = (a2 * b2).sum(-1)
        cos2 = ab2 / (a2l1 * b2l1 + eps)
        sin2 = (1 - cos2.pow(2) + eps).sqrt()
        c2 = a2 * (ab2 / (a2l2 + eps))[:, :, None]
        cb2 = b2 - c2
        cb2l1 = b2l1 * sin2

        cos = (cb1 * cb2).sum(-1) / (cb1l1 * cb2l1 + eps)

        dims = tuple(range(cos.ndimension())[1:])
        loss = (cos + 1).pow(2).sum(dims)
        if self.average:
            return loss.sum() / batch_size
        else:
            return loss


class ARAPLoss(nn.Module):
    def __init__(self, vertex, faces, average=False, size_factor=1.0):
        super(ARAPLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        self.size_factor = size_factor
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

        self.laplacian_sp = self.laplacian.clone().to_sparse().cuda()


    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3


        x = x * self.size_factor
        dx = dx * self.size_factor

        nv_num =  x.shape[0]
        diffx = torch.zeros(self.laplacian.bool().sum()).cuda()
        diffdx = torch.zeros(self.laplacian.bool().sum()).cuda()


        for i in range(3):
            dx_sub = torch.sparse.mm(self.laplacian_sp, torch.diag_embed(dx[ :, i]))
            dx_diff = (dx_sub - dx[:, i:i + 1])[self.laplacian.bool()]

            x_sub = torch.sparse.mm(self.laplacian_sp, torch.diag_embed(x[:, i]))
            x_diff = (x_sub - x[:, i:i + 1])[self.laplacian.bool()]

            diffdx = torch.add(diffdx, (dx_diff).pow(2))
            diffx = torch.add(diffx, (x_diff).pow(2))


        diff = (diffx - diffdx).abs().mean()


        return diff
