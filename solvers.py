import logging
import numpy as np
from bc import bcsum

__all__ = ['solve']

def solve(A,b,u, *constriants, **kwargs):
    bc_method = kwargs.get('bc_method','default')
    method = kwargs.get('method','direct')

    #make sure forces and boundary conditions not applied
    #to same node
    ok = 0
    errors = 0
    neumann = []
    dirichlet = []
    constrianted_nodes = []
    for constriant in constriants:
        if constriant.type == 'dirichlet':
            ok = 1
            dirichlet.append(constriant)
        else:
            neumann.append(constriant)
        if np.any(np.in1d(constriant.nodes,constrianted_nodes)):
            logging.error('multiple BCs applied to same node')
            errors += 1
        constrianted_nodes.extend(constriant.nodes)
    methods = ('direct',)
    if method not in methods:
        a = ', '.join(methods)
        logging.error('expected method to be one of {0},got {1}'.format(a,method))
        errors += 1

    #make sure the the system is stable
    if not ok:
        logging.error('system requires at least one prescribed displament'
                      'to be stable')
        errors += 1

    if errors:
        raise  SystemExit('stopping due to previous errors')

    if method == 'direct':
        return linear_solve(A,b,u,dirichlet,neumann,bc_method)
def linear_solve(A,b,u,dirichlet,neumann,bc_method):
    #Apply boundary conditions on copies of A adn b that we retain A and b
    #for later use.
    Abc = np.array(A)
    bbc = np.array(b)

    X = 1.e9 * np.amax(A)
    for (node, dof, magnitude) in bcsum(neumann):
        i = u.V.mesh.dof_map(node) * u.V.num_dof_per_node + dof
        bbc[i] = magnitude

    for (node,dof,magnitude) in bcsum(dirichlet):
        i = u.V.mesh.dof_map(node) * u.V.num_dof_per_node + dof
        if bc_method == 'default':
            # Default method - apply bcs such that the global stiffness
            # remains symmetric
            #Modify the RHS
            bbc -= [Abc[j,i] * magnitude for j in range(u.V.num_dof)]
            bbc[i] = magnitude
            #Modify the stiffness
            Abc[i,:] = 0
            Abc[:,i] = 0
            Abc[i,i] = 1

        elif bc_method == 'penalty':
            raise  NotImplementedError('bc_method=="penalty')

        else:
            raise  ValueError('unknown bc_method {0}'.format(bc_method))

    u += np.linalg.solve(Abc,bbc)
    r = np.dot(A,u.vector) - b
    return


