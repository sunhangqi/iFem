import numpy as np

class GaussLegendreShapeFunction(object):
    num_points = None
    num_coord  = None

class LinearGaussLegend1D(GaussLegendreShapeFunction):
    num_points = 2
    num_coord = 1
    def eval(self,xi):
        '''
        Value of linear 1D shape function evaluated at xi
        :param xi: real
            Point inside element in natural coordinates
        :return:
            f:ndarray of real (2,)
              The element shape function array at xi
        '''
        return np.array([1. - xi, 1. + xi]) * (xi >= -1.) * (xi <= 1.) / 2.

    def grad(self,xi):
        '''
        Devibative of linear 1D shape functions
        :param xi: real
            Point inside element in natural coordinates
        :return:
            df:narray of real (1,2)
            The element shape function derivative array at xi
        '''
        return np.array([-1.,1.]) * (xi >= -1.) * (xi <= 1.) / 2.