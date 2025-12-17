import numpy as np

from ..element_hcurl import ElementHcurl
from ...refdom import RefTri
from ..discrete_field import DiscreteField


class ElementTriN3(ElementHcurl):
    """The third order Nedelec element."""

    facet_dofs = 3
    interior_dofs = 6
    maxdeg = 3
    dofnames = ['u^t'] * 3 + ['u^x', 'u^y'] * 6
    doflocs = np.array([
            [0.25, 0.0],
            [0.50, 0.0],
            [0.75, 0.0],
            [0.75, 0.25],
            [0.50, 0.50],
            [0.25, 0.75],
            [0.0, 0.75],
            [0.0, 0.50],
            [0.0, 0.25],
            [0.25, 0.25],
            [0.25, 0.25],
            [0.50, 0.25],
            [0.50, 0.25],
            [0.25, 0.50],
            [0.25, 0.50]
        ])

    refdom = RefTri

    def gbasis(self, mapping, X, i, tind=None):
        """Covariant Piola transformation.
        Overridden to allow for higher order"""
        orient = self.orient(mapping, i, tind)
        target_swap = i
        if i < 9:
            edge_idx = i // 3
            local_idx = i % 3
            if edge_idx in [0, 1]:
                if local_idx == 0:
                    target_swap = i + 2
                elif local_idx == 2:
                    target_swap = i - 2
                swap_condition = -1  # Swap if orient < 0
            elif edge_idx == 2:
                if local_idx == 0:
                    target_swap = i + 2
                elif local_idx == 2:
                    target_swap = i - 2
                swap_condition = 1  # Swap if orient > 0

        def get_lbasis_fixed(idx):
            p, dp = self.lbasis(X, idx)
            if 6 <= idx <= 8:
                # Edge 2 vectors point down in definition,
                # but reference points Up.
                return -p, -dp
            return p, dp

        if target_swap == i:
            phi, dphi = get_lbasis_fixed(i)

            invDF = mapping.invDF(X, tind)
            detDF = mapping.detDF(X, tind)

            val_final = np.einsum('ijkl,il,k->jkl', invDF, phi, orient)
            curl_final = dphi / detDF * orient[:, None]

        else:
            phi_A, dphi_A = get_lbasis_fixed(i)
            phi_B, dphi_B = get_lbasis_fixed(target_swap)

            invDF = mapping.invDF(X, tind)
            detDF = mapping.detDF(X, tind)

            val_A = np.einsum('ijkl,il,k->jkl', invDF, phi_A, orient)
            curl_A = dphi_A / detDF * orient[:, None]

            val_B = np.einsum('ijkl,il,k->jkl', invDF, phi_B, orient)
            curl_B = dphi_B / detDF * orient[:, None]

            if swap_condition == -1:
                mask = (orient > 0).astype(np.float64)
            else:
                mask = (orient < 0).astype(np.float64)

            mask_val = mask[None, :, None]
            mask_curl = mask[:, None]

            val_final = val_A * mask_val + val_B * (1.0 - mask_val)
            curl_final = curl_A * mask_curl + curl_B * (1.0 - mask_curl)
        return (DiscreteField(value=val_final, curl=curl_final),)

    def lbasis(self, X, i):
        x, y = X

        x2 = x**2
        y2 = y**2
        xy = x * y

        if i == 0:
            phi = np.array([
                -8.0*x2*y + 8.0*x2 - 16.0*x*y2 + 26.0*xy -
                10.0*x - 8.0*y**3 + 18.0*y2 - 13.0*y + 3.0,
                x*(8.0*x2 + 16.0*xy - 10.0*x + 8.0*y2 - 10.0*y + 3.0)
            ])
            dphi = 32.0*x2 + 64.0*xy - 46.0*x + 32.0*y2 - 46.0*y + 16.0

        elif i == 1:
            phi = np.array([
                16.0*x2*y - 16.0*x2 + 16.0*x*y2 - 32.0*xy + 16.0*x
                - 4.0*y2 + 7.0*y - 3.0,
                x*(-16.0*x2 - 16.0*xy + 16.0*x + 4.0*y - 3.0)
            ])
            dphi = -64.0*x2 - 64.0*xy + 64.0*x + 12.0*y - 10.0

        elif i == 2:
            phi = np.array([
                -8.0*x2*y + 8.0*x2 + 6.0*xy - 6.0*x - 1.0*y + 1.0,
                x*(8.0*x2 - 6.0*x + 1.0)
            ])
            dphi = 32.0*x2 - 18.0*x + 2.0

        elif i == 3:
            phi = np.array([
                y*(-8.0*x2 + 6.0*x - 1.0),
                x*(8.0*x2 - 6.0*x + 1.0)
            ])
            dphi = 32.0*x2 - 18.0*x + 2.0

        elif i == 4:
            phi = np.array([
                y*(-16.0*xy + 4.0*x + 4.0*y - 1.0),
                x*(16.0*xy - 4.0*x - 4.0*y + 1.0)
            ])
            dphi = 64.0*xy - 12.0*x - 12.0*y + 2.0

        elif i == 5:
            phi = np.array([
                y*(-8.0*y2 + 6.0*y - 1.0),
                x*(8.0*y2 - 6.0*y + 1.0)
            ])
            dphi = 32.0*y2 - 18.0*y + 2.0

        elif i == 6:
            phi = np.array([
                y*(-8.0*y2 + 6.0*y - 1.0),
                8.0*x*y2 - 6.0*xy + 1.0*x - 8.0*y2 + 6.0*y - 1.0
            ])
            dphi = 32.0*y2 - 18.0*y + 2.0

        elif i == 7:
            phi = np.array([
                y*(16.0*xy - 4.0*x + 16.0*y2 - 16.0*y + 3.0),
                -16.0*x2*y + 4.0*x2 - 16.0*x*y2 + 32.0*xy
                - 7.0*x + 16.0*y2 - 16.0*y + 3.0
            ])
            dphi = -64.0*xy + 12.0*x - 64.0*y2 + 64.0*y - 10.0

        elif i == 8:
            phi = np.array([
                y*(-8.0*x2 - 16.0*xy + 10.0*x - 8.0*y2 + 10.0*y - 3.0),
                8.0*x**3 + 16.0*x2*y - 18.0*x2 + 8.0*x*y2 -
                26.0*xy + 13.0*x - 8.0*y2 + 10.0*y - 3.0
            ])
            dphi = 32.0*x2 + 64.0*xy - 46.0*x + 32.0*y2 - 46.0*y + 16.0

        elif i == 9:
            phi = np.array([
                y*(8.0*x2 + 32.0*xy - 30.0*x + 24.0*y2 - 42.0*y + 18.0),
                x*(-8.0*x2 - 32.0*xy + 14.0*x - 24.0*y2 + 26.0*y - 6.0)
            ])
            dphi = -32.0*x2 - 128.0*xy + 58.0*x - 96.0*y2 + 110.0*y - 24.0

        elif i == 10:
            phi = np.array([
                y*(-24.0*x2 - 32.0*xy + 26.0*x - 8.0*y2 + 14.0*y - 6.0),
                x*(24.0*x2 + 32.0*xy - 42.0*x + 8.0*y2 - 30.0*y + 18.0)
            ])
            dphi = 96.0*x2 + 128.0*xy - 110.0*x + 32.0*y2 - 58.0*y + 24.0

        elif i == 11:
            phi = np.array([
                y*(-16.0*x2 - 32.0*xy + 36.0*x + 8.0*y - 8.0),
                x*(16.0*x2 + 32.0*xy - 20.0*x - 8.0*y + 4.0)
            ])
            dphi = 64.0*x2 + 128.0*xy - 76.0*x - 24.0*y + 12.0

        elif i == 12:
            phi = np.array([
                y*(24.0*x2 + 16.0*xy - 22.0*x - 4.0*y + 4.0),
                x*(-24.0*x2 - 16.0*xy + 30.0*x + 4.0*y - 6.0)
            ])
            dphi = -96.0*x2 - 64.0*xy + 82.0*x + 12.0*y - 10.0

        elif i == 13:
            phi = np.array([
                y*(-16.0*xy + 4.0*x - 24.0*y2 + 30.0*y - 6.0),
                x*(16.0*xy - 4.0*x + 24.0*y2 - 22.0*y + 4.0)
            ])
            dphi = 64.0*xy - 12.0*x + 96.0*y2 - 82.0*y + 10.0

        elif i == 14:
            phi = np.array([
                y*(32.0*xy - 8.0*x + 16.0*y2 - 20.0*y + 4.0),
                x*(-32.0*xy + 8.0*x - 16.0*y2 + 36.0*y - 8.0)
            ])
            dphi = -128.0*xy + 24.0*x - 64.0*y2 + 76.0*y - 12.0

        else:
            self._index_error()

        return phi, dphi
