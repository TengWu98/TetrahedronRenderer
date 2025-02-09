from typing import NamedTuple
import torch
import torch.nn as nn
from . import _C

'''
===============================================================================
* TetRenderer

Renderer for faces in a compact set of tetrahedra.
Efficient, support accurate depth testing, but only propagate gradients to opacities.
===============================================================================
'''

class TetRenderSettings(NamedTuple):
    image_height: int
    image_width: int
    bg : torch.Tensor
    ray_random_seed : int

def render_tet(
        verts: torch.Tensor,
        faces: torch.Tensor,
        verts_color: torch.Tensor,
        faces_opacity: torch.Tensor,

        mv_mats: torch.Tensor,
        proj_mats: torch.Tensor,
        verts_depth: torch.Tensor,
        faces_intense: torch.Tensor,

        tets: torch.Tensor,
        face_tets: torch.Tensor,
        tet_faces: torch.Tensor,

        render_settings: TetRenderSettings
):
    return _RenderTet.apply(
        verts,
        faces,
        verts_color,
        faces_opacity,

        mv_mats,
        proj_mats,
        verts_depth,
        faces_intense,

        tets,
        face_tets,
        tet_faces,
        render_settings
    )

class _RenderTet(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,

            verts,
            faces,
            verts_color,
            faces_opacity,

            mv_mats,
            proj_mats,
            verts_depth,
            faces_intense,

            tets,
            face_tets,
            tet_faces,

            render_settings: TetRenderSettings
    ):
        inv_mv_mats = torch.inverse(mv_mats)
        inv_proj_mats = torch.inverse(proj_mats)

        # Restructure arguments the way that the C++ lib expects them
        args = (
            render_settings.bg,

            verts,
            faces,
            verts_color,
            faces_opacity,

            mv_mats,
            proj_mats,
            inv_mv_mats,
            inv_proj_mats,
            verts_depth,
            faces_intense,

            tets,
            face_tets,
            tet_faces,

            render_settings.image_height,
            render_settings.image_width,
            render_settings.ray_random_seed
        )

        # Invoke C++/CUDA renderer
        try:
            color, depth, active, pointBuffer, faceBuffer, binningBuffer, imgBuffer = _C.render_tets(*args)
        except Exception as ex:
            print("\nAn error occured in forward.")
            raise ex

        active = (active > 0.5) # change to boolean tensor;

        # Keep relevant tensors for backward
        ctx.render_settings = render_settings
        ctx.save_for_backward(
            verts,
            faces,
            verts_color,
            faces_opacity,

            mv_mats,
            proj_mats,
            inv_mv_mats,
            inv_proj_mats,
            verts_depth,
            faces_intense,

            tets,
            face_tets,
            tet_faces,

            pointBuffer,
            faceBuffer,
            binningBuffer,
            imgBuffer)

        return color, depth, active

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, grad_out_active):

        # Restore necessary values from context
        render_settings = ctx.render_settings
        verts, faces, verts_color, faces_opacity, \
            mv_mats, proj_mats, inv_mv_mats, inv_proj_mats, \
            verts_depth, faces_intense, \
            tets, face_tets, tet_faces, \
            pointBuffer, faceBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (render_settings.bg,

                verts,
                faces,
                verts_color,
                faces_opacity,

                mv_mats,
                proj_mats,
                inv_mv_mats,
                inv_proj_mats,
                verts_depth,
                faces_intense,

                tets,
                face_tets,
                tet_faces,

                grad_out_color,
                grad_out_depth,

                pointBuffer,
                faceBuffer,
                binningBuffer,
                imgBuffer)

        # Compute gradients for relevant tensors by invoking backward method
        try:
            grad_results = _C.render_tets_backward(*args)
            grad_verts_color, grad_faces_opacity = grad_results[0], grad_results[1]
        except Exception as ex:
            print("\nAn error occured in backward.\n")
            raise ex

        grads = (
            None,
            None,
            grad_verts_color,
            grad_faces_opacity,

            None,
            None,
            None,
            None,

            None,
            None,
            None,

            None)

        return grads

class TetRenderer(nn.Module):
    def __init__(self, render_settings: TetRenderSettings):
        super().__init__()
        self.render_settings = render_settings

    def forward(self,
                verts: torch.Tensor,
                faces: torch.Tensor,
                verts_color: torch.Tensor,
                faces_opacity: torch.Tensor,

                mv_mats: torch.Tensor,
                proj_mats: torch.Tensor,
                verts_depth: torch.Tensor,
                faces_intense: torch.Tensor,

                tets: torch.Tensor,
                face_tets: torch.Tensor,
                tet_faces: torch.Tensor):

        '''
        Gradients are only provided for [verts_color] and [faces_opacity].
        @ Note: [verts_depth] is not used, because depth is computed using
        normalization based on w coords, which is non linear.

        @ verts: [# vertex, 3], positions of vertices
        @ faces: [# face, 3], indices of vertices of faces
        @ verts_color: [# vertex, 3], colors of vertices
        @ faces_opacity: [# face,], opacity of faces
        
        @ mv_mats: [# batch, 4, 4], batch of modelview matrices
        @ proj_mats: [# batch, 4, 4], batch of projection matrices
        @ verts_depth: [# batch, # vertex,], depth of vertices
        @ faces_intense: [# batch, # face,], intensity of faces
        
        @ tets: [# tet, 4], indices of vertices of tets
        @ face_tets: [# face, 2], indices of tets that a face belongs to
        @ tet_faces: [# tet, 4], indices of faces that a tet owns
        '''

        render_settings = self.render_settings

        # reshape;

        color, depth, active = render_tet(
            verts.to(dtype=torch.float32),
            faces.to(dtype=torch.int32),
            verts_color.to(dtype=torch.float32),
            faces_opacity.to(dtype=th.float32),

            mv_mats.to(dtype=torch.float32).transpose(1, 2),
            proj_mats.to(dtype=torch.float32).transpose(1, 2),
            verts_depth.to(dtype=torch.float32),
            faces_intense.to(dtype=torch.float32),

            tets.to(dtype=torch.int32),
            face_tets.to(dtype=torch.int32),
            tet_faces.to(dtype=torch.int32),

            render_settings
        )

        return color, depth, active