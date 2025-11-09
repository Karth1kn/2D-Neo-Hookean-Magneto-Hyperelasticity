import numpy as np
import matplotlib.pyplot as plt 
from D2_Planar import val_array, con_array, edge_array, force_list, bc_lst, timer
import matplotlib.patches as patches
from scipy.linalg import solveh_banded, solve_banded
import scipy.sparse.linalg
import math
import time
np.set_printoptions(suppress=True, precision=1, linewidth=200)
#from dolfin import *
#from arc_length.force_control_solver import force_control 

def deform(x,theta, s):
    theta = np.radians(theta)
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])
    S = np.array([[s,0],
                [0,s]])
    return Q@S@x
######## CONSTANTS ########
g = 0 # Acceleration due to gravity
rho = 1 # Density of the material
mu = 10
lamb = 4
gload = 0

E = 1 # N/mm**2
nu = 0.3
mu = E/(2*(1+nu))
lamb = (E*nu)/((1+nu)*(1-2*nu))

mu0 = 4*np.pi*1e-6

l = 0 #load factor
T0 = 1 #Thickness of the material
r3 = 1/math.sqrt(3) # 1/sqrt(3) for the quadrature points in the parametric space
i2 = np.eye(2,2) #Second order identity tensor
I0 = np.einsum('ij,kl->ijkl',i2,i2)
I1 = np.einsum('ik,jl->ijkl',i2,i2) # Fourth order identity tensor
I2 = np.einsum('il,jk->ijkl',i2,i2) # fourth order transposition tensor
###########################
cord = val_array
lc = len(cord) # Number of nodes
con = con_array
#print(con)
le = len(con) # Number of elements


fig, ax = plt.subplots()
ax.set_facecolor('black')
plt.axis('equal')

def plotter(cord):
    for i,j in enumerate(con):
        quadr = np.array([cord[j[0]], cord[j[1]], cord[j[2]], cord[j[3]]])
        #plt.text(np.sum(quadr[:,0])/4, np.sum(quadr[:,1])/4, s=f'{i}', color='green', fontsize=15, fontweight='bold')
        fos= 1
        r,g,b,a = 1, 1, 0.3,  1
        ax.fill(quadr[:,0], quadr[:,1], color=(r,g,b,a) )
        triangle = patches.Polygon(quadr, closed=True, color = (0.5,0.5,0.5), alpha=1)
        
        ax.add_patch(triangle)
    for i,k in enumerate(cord):
        continue
        plt.scatter(k[0], k[1], color='red', s=50)
        plt.text(k[0], k[1], s=f'{i}', color='black', fontsize=10, fontweight='bold')
    
    for i,j in enumerate(edge_array):
        break
        #plt.plot([cord[j[0],0], cord[j[1],0]], [cord[j[0],1], cord[j[1],1]], r='black', linewidth=2)
        #plt.text((cord[j[0],0] + cord[j[1],0])/2, (cord[j[0],1] + cord[j[1],1])/2, s=f'{i}', color='black', fontsize=8, fontweight='bold')

    plt.axis("equal")

Fbody = np.zeros((2*len(cord), 1)) # Body force vector
Ftraction = np.zeros((2*len(cord), 1)) # Traction force vector

Xa = cord.T # Material coordinates
xa = np.copy(Xa) # Spatial coordinates
#xa = deform(np.copy(cord.T), 30,3) # Spatial coordinates
va = np.ones((2, len(cord))) # Velocity vector
#ua = np.ones((2, len(cord))) # Displacement vector



A0 = np.array([np.linalg.norm(np.cross(Xa[:,con[ele,3]]-Xa[:,con[ele,0]], Xa[:,con[ele,0]]-Xa[:,con[ele,1]])) for ele in range(len(con))]) # Initial area of the elements
#a = np.array([np.linalg.norm(np.cross(xa[:,con[ele,3]]-xa[:,con[ele,0]], xa[:,con[ele,0]]-xa[:,con[ele,1]])) for ele in range(len(con))]) # Current area of the elements

gload = lambda i: A0[i]*T0*rho*g # Total mass of the system


#interp = lambda e,n,ele, xs: 0.25*( (1-e)*(1-n)*xs[:,ele[0]] + (1+e)*(1-n)*xs[:,ele[1]] + (1+e)*(1+n)*xs[:,ele[2]] + (1-e)*(1+n)*xs[:,ele[3]] ) # Bilinear interpolation (shape) function
#interp = lambda e,n,ele, xs: 0.25*np.einsum('ij,j->i', xs[:,ele], np.array([(1-e)*(1-n), (1+e)*(1-n), (1+e)*(1+n), (1-e)*(1+n)]))
shapefn = lambda e,n: 0.25*np.array([(1-e)*(1-n), (1+e)*(1-n), (1+e)*(1+n), (1-e)*(1+n)]) # Shape function coefficients for bilinear interpolation
interp = lambda e,n,ele, xs: 0.25*(xs[:,ele]@np.array([[(1-e)*(1-n), (1+e)*(1-n), (1+e)*(1+n), (1-e)*(1+n)]]).T)[:,0] # Bilinear interpolation (shape) function using einsum

#dinterpE = lambda n,ele, xs: 0.25*( -(1-n)*xs[:,ele[0]] + (1-n)*xs[:,ele[1]] + (1+n)*xs[:,ele[2]] + -(1+n)*xs[:,ele[3]] ) # Derivative of interpolation function with respect to e
dinterpE = lambda e,n,ele, xs: 0.25*(xs[:,ele]@np.array([[ -(1-n), (1-n), (1+n), -(1+n)]]).T)[:,0] # Derivative of interpolation function with respect to e (1D ndarray)

#dinterpN = lambda e,ele, xs: 0.25*( -(1-e)*xs[:,ele[0]] + -(1+e)*xs[:,ele[1]] + (1+e)*xs[:,ele[2]] + (1-e)*xs[:,ele[3]] ) # Derivative of interpolation function with respect to n
dinterpN = lambda e,n,ele, xs: 0.25*(xs[:,ele]@np.array([[ -(1-e), -(1+e), (1+e), (1-e)]]).T)[:,0] # Derivative of interpolation function with respect to n (1D ndarray)

X = lambda e,n, ele: interp(e,n,ele,Xa) # Material coordinates
x = lambda e,n,ele: interp(e,n,ele,xa) # Spatial coordinates
#u = lambda e,n,ele: interp(e,n,ele,ua) # Displacement vector

dNaden = lambda e,n:  0.25*np.array([[-(1-n), (1-n), (1+n), -(1+n)], #dN/de
                                     [-(1-e), -(1+e), (1+e), (1-e)]]) #dN/dn

JXen = lambda e,n, ele: np.linalg.det(np.column_stack((dinterpE(e,n,ele,Xa), dinterpN(e,n,ele,Xa)))) # Jacobian determinant for material coordinates
Jxen = lambda e,n, ele: np.linalg.det(np.column_stack((dinterpE(e,n,ele,xa), dinterpN(e,n,ele,xa)))) # Jacobian determinant for spatial coordinates

# dXde = dinterpE(n,ele,Xa), dXdn = dinterpN(e,ele,Xa), dXen = np.column_stack((dinterpE(n,ele,Xa), dinterpN(e,ele,Xa))) # Derivative of interpolation function with respect to e and n for material coordinates
# dxde = dinterpE(n,ele,xa), dXdn = dinterpN(e,ele,xa), dXen = np.column_stack((dinterpE(n,ele,xa), dinterpN(e,ele,xa))) # Derivative of interpolation function with respect to e and n for spatial coordinates
#dNadX = lambda e,n,ele: np.einsum('IK,KJ->IJ', np.linalg.inv(np.column_stack((dinterpE(e,n,ele,Xa), dinterpN(e,n,ele,Xa)))).T, dNaden(e,n)) #(dX/de)-T@dNade
#dNadx = lambda e,n,ele: np.einsum('IK,KJ->IJ', np.linalg.inv(np.column_stack((dinterpE(e,n,ele,xa), dinterpN(e,n,ele,xa)))).T, dNaden(e,n)) #(dx/de)-T@dNade

dNadxX = lambda e,n,ele,x: np.einsum('IK,KJ->IJ', np.linalg.inv(np.column_stack((dinterpE(e,n,ele,x), dinterpN(e,n,ele,x)))).T, dNaden(e,n))
#print(np.linalg.norm(np.cross(Xa[:,con[0,3]]-Xa[:,con[0,0]], Xa[:,con[0,0]]-Xa[:,con[0,1]]))) # Area of the triangle formed by the first three nodes of the element
#print(3*np.linalg.norm(np.cross(Xa[:,con[0,1]]-Xa[:,con[0,2]], Xa[:,con[0,3]]-Xa[:,con[0,2]])))
def Kinematics(e, n, ele, xa, Xa):
    Na = shapefn(e,n)
    F = np.zeros((2, 2))
    xi = xa[:, ele].T
    #vi = va[:, ele].T
    #ui = ua[:, ele].T
    #d = np.zeros((2, 2))
    #em = np.zeros((2, 2))
    dNadX1 = dNadxX(e, n, ele, Xa)  # Derivative of the shape function with respect to e and n for material coordinates
    dNadx1 = dNadxX(e, n, ele, xa)
    for i in range(4):
        break
        #L = np.einsum('i,J->iJ', vi[i,:], dNadx1[:,i]) # velocity gradient tensor
        #d += 0.5*(L + L.T) # Rate of deformation tensor
        F += np.einsum('i,J->iJ', xi[i,:], dNadX1[:,i])
        #Ee1 = np.einsum('i,J->iJ', ui[i,:], dNadx1[:,i]) 
        #em += 0.5*(Ee1 + Ee1.T) # Small strain tensor
        #F += np.outer( xi[i,:], dNadX(e, n, ele)[:,i])
    F =  (dNadX1@xi).T
    J = abs(np.linalg.det(F))  # Jacobian determinant
    #print(F(0.5, 0.5, con[0]))
    C = np.einsum('kI,kJ->IJ', F, F)  # Right Cauchy-Green deformation tensor
    #C1 = F(e, n, ele).T @ F(e, n, ele)
    b = np.einsum('iK,jK->ij', F, F) # Left Cauchy-Green deformation tensor
    #E = 0.5 * (C - i2) # Green-Lagrange strain tensor
    #ee = 0.5 * (i2 - np.linalg.inv(b)) # Euler-Almansi strain tensor

    l, Ni = np.linalg.eig(C)  # Eigenvalues and eigenvectors
    U = np.zeros((2, 2))
    for i, val in enumerate(l):
        U += (val**0.5)*np.outer(Ni[:, i], Ni[:, i])
        U += (val**0.5)*np.einsum('i,j->ij', Ni[:, i], Ni[:, i])
    # ROTATION TENSOR
    R = F@np.linalg.inv(U)

    #A00 = A0[idx] # Reference area of the undeformed element
    #a = np.linalg.norm(np.cross(xa[:,con[idx,3]]-xa[:,con[idx,0]], xa[:,con[idx,0]]-xa[:,con[idx,1]])) # current area of the deformed element
    #Jxen(e, n, ele) # Jacobian determinant for spatial coordinates
    #t = J*A00*T0/a

    lamb_ = lamb/J
    mu_ = (mu - lamb*np.log(J))/J
    C4 = lamb_*I0 + mu_*(I1 + I2) # Fourth order elasticity tensor
    
    return J, Jxen(e, n, ele), b, t, dNadx1, C4,F, R



def total_tangent_matrix(dNCdN, dNCaudN, ele, cons):
    for a in range(4):
        for b in range(4):
            i,j = ele[a], ele[b]  
            Kc[2*i:2*i+2, 2*j:2*j+2] += dNCdN[:,:,a,b]*cons
            Ks[2*i:2*i+2, 2*j:2*j+2] += (dNCaudN[a,b]*cons)*i2



def stress_matrix(dNadx, Cau, ele):
    for i in range(4):
        a, b = ele[i], ele[(i+1) % 4]  # Pair of nodes in the element
        dNa,dNb = dNadx[:,i], dNadx[:,(i+1) % 4] # Derivatives of the shape functions with respect to e and n for the pair of nodes
        dNaCaudNb = np.einsum('i,ij,j->', dNa, Cau, dNb)*i2
        Ks[2*a:2*a+2, 2*b:2*b+2] += dNaCaudNb

        dNbCaudNa = np.einsum('i,ij,j->', dNb, Cau, dNa)*i2
        Ks[2*b:2*b+2, 2*a:2*a+2] += dNbCaudNa

Cau = np.zeros((len(con),2,2))

Br = 1e-4*np.array([1,0]) # T (REMANENT FIELD INSIDE ELASTOMER)
Ba = 6.0e-4*np.array([0,1]) # T (APPLIED MAGNETIC FIELD)

#print(dNadx(0.5, 0.5, con[0]))
qp = np.array([[r3, r3], [r3, -r3], [-r3, -r3], [-r3, r3]])  # Quadrature points in the parametric space

def Builder(ele, eleidx, xa, Xa): # Internal forces of all the nodes in the element
    ''' Internal force vector T = [[T1x, T2x, T3x, T4x], 
                                   [T1y, T2y, T3y, T4y]] for each node in the element '''
    def Quad(e, n): # Common function for all forces and stiffness matrices
        J,Jxen,b,t, dNadx,C,F, R = Kinematics(e, n, ele, xa, Xa)
        dNCdN = np.einsum('ka,ikjl,lb->ijab', dNadx, C, dNadx)  # Derivative of the shape function with respect to e and n
        CauU = (mu/J)*(b - i2) + (lamb/J)*np.log(J)*i2    # Cauchy stress tensor
        
        Fbr = np.einsum('ij,j->i', R,Br)
        CauM = (1/(mu0*J)) * np.einsum('i,j->ij', Ba, Fbr)
        Cau = CauU - CauM
        dNCaudN = np.einsum('ia,ij,jb->ab', dNadx, Cau, dNadx)  # Derivative of the shape function with respect to e and n for the stress part

        #print(Cau)
        Na = interp(e, n, ele, xa)  # Shape function values at the quadrature point
        T = Cau @ dNadx
        cons = abs(Jxen)*T0
        #total_tangent_matrix(dNadx, C, Cau, ele, cons)
        total_tangent_matrix(dNCdN, dNCaudN, ele, cons) 
        #stress_matrix(dNadx, Cau, ele)              
        #T = np.einsum('ij,j->i', Cau, dNadx1[:,node])  # Internal force vector

        return cons*T, gload(eleidx)*cons*shapefn(e,n), Cau # Return the internal force vector and external force for the element and at the given parametric coordinates e and n
    fint = np.zeros((2, 4))  # Internal force vector for the element
    fext = np.zeros((1, 4))[0,:]  # External force vector for the element
    cau = np.zeros((2, 2)) 
    for i in qp:
        ii, ee, Caui = Quad(*i)
        fint += ii
        fext += ee
        cau += Caui
    Cau[eleidx,:,:] = cau/4 # Store the Cauchy stress tensor for the element

    return fint, fext #Quad(-r3, -r3)[0] + Quad(r3, -r3)[0] + Quad(r3, r3)[0] + Quad(-r3, r3)[0]  # Return the internal force vector for the element at the given parametric coordinates e and n

             

t = time.time()
@timer
def internal_forces(xa, Xa):
    T = np.zeros((2*len(cord), 1)) # Internal force vector
    Fbody = np.zeros((2*len(cord), 1)) # Body force vector
    #### Main loop for Internal Force Vector Assembly ####
    for i,ele in enumerate(con):
        fi, fe = Builder(ele, i, xa, Xa)
        fi = fi.T
        T[2*ele,:] += np.array([fi[:,0]]).T # Sum the internal forces of all the elements in x direction
        T[2*ele+1,:] += np.array([fi[:,1]]).T # Sum the internal forces of all the elements in y direction
        #Fbody[2*ele+1,:] += np.array([fe]).T # Sum the body forces of all the elements in x direction

    return T, Fbody



for key, val in force_list.items():
    Ftraction[2*key] = val[0]
    Ftraction[2*key+1] = val[1]



def bc(mat, bc_list):
    mat[bc_list,:] = 0
    if len(mat[0])!=1:
        mat[:,bc_list] = 0
        mat[bc_list,bc_list] = 1
    return mat

s = time.time()
print("Time taken for Internal Force Vector Assembly:", s-t)

@timer
def color_plotter(cord, stress):
    for i,j in enumerate(con):
        quadr = np.array([cord[j[0]], cord[j[1]], cord[j[2]], cord[j[3]]])

        #Von mises
        
        fos= stress[i]/np.max(stress)
        if fos**2>=1:
            r,g,b,a = 0 ,1 ,0 ,1
            plt.annotate('material will fail at this load',(3000,-5000),color= 'white')
            ax.fill(quadr[:,0], quadr[:,1], color=(r,g,b,a) )

        elif fos<=0:
            r,g,b,a = 1+fos, 1+fos, 1,  1
            ax.fill(quadr[:,0], quadr[:,1], color=(r,g,b,a) )
        elif fos>0:
            r,g,b,a= 1, 1-fos, 1-fos,  1         
            ax.fill(quadr[:,0], quadr[:,1], color=(r,g,b,a) )


#bandwidth = max([max(i)-min(i) for i in con ])*2+2


# NEWTON RAPHSON METHOD
Kc = np.zeros((2*lc, 2*lc)) # Constitutive part of Stiffness matrix
Ks = np.zeros((2*lc, 2*lc)) # Stress part of Stiffness matrix
K = Kc+Ks
T, Fbody = internal_forces(xa, Xa) # Internal force vector
#K = Kc + Ks # Total stiffness matrix
for key, val in force_list.items():
    Ftraction[2*key] = val[0]
    Ftraction[2*key+1] = val[1]

F = Ftraction #+ Fbody # Total force vector
#fig1, ax1 = plt.subplots()
plt.ion()
conv = 0
convprev = conv
for i in range(25):
    #break
    print('Iteration : ', i)
    Kc = np.zeros((2*lc, 2*lc)) # Constitutive part of Stiffness matrix
    Ks = np.zeros((2*lc, 2*lc))

    T, Fbody = internal_forces(xa, Xa)
    K = Kc + Ks  # Update the stiffness matrix
    R = -bc(T, bc_lst) + F  # Residual force vector
    conv = np.linalg.norm(T)
    print('Residual: ',conv)
    if abs(conv - convprev)/conv < 1e-3:
        print("Convergence achieved")
        break
    else:
        convprev = conv
    #print(T,'r')
    t1 = time.time()
    Kbc = bc(K, bc_lst)  # Apply boundary conditions to the stiffness matrix
    Rbc = bc(R, bc_lst)  # Apply boundary conditions to the residual force vector
    t2 = time.time()

    #diagonals = np.array([np.append(np.diag(Kbc, -i), np.zeros([1,i])[0]) for i in range(bandwidth)  ])
    #u = solveh_banded(diagonals, Rbc, lower= True)
    u = scipy.sparse.linalg.spsolve(Kbc,Rbc) 
    #U = scipy.sparse.linalg.spsolve(Kbc, Rbc)
    t3 = time.time()
    #print("Time taken for solving the system of equations by sparse solver:", t3-t2)
    #u = np.linalg.inv(bc(K, bc_lst)) @ bc(R, bc_lst)  # Solve the system of equations
    #u = scipy.sparse.linalg.splu(Kbc).solve(Rbc)
    t4 = time.time()
    #print("Time taken for solving the system of equations by super LU method:", t4-t3)

    ua = (u.reshape((lc,2))).T  # Update the displacement vector
    xa += ua
    ax.clear()
    #ax1.scatter(xa[1,2176]-Xa[1,2176], R[2176*2+1,0])
    ##plt.scatter(0,xa[1,2176], color='green', s=50)
    #s = Cau - (1/2) * np.trace(Cau) * i2
    #von_mises = np.sqrt((3/2) * np.einsum('ijk,ijk->i', s,s))
    color_plotter(xa.T, Cau[:,0,0])
    plt.pause(0.01)
    #print(np.linalg.norm(u))



plotter(Xa.T)



color_plotter(xa.T, Cau[:,0,0])


plt.ioff()  # Turn off interactive mode

def sparser():
    fig,ax = plt.subplots()
    oy = []
    for q in range(len(cord)):
        for w in range(len(cord)):
            if not np.array_equal(K[2*q:2*q+2 , 2*w:2*w+2] , np.zeros([2,2])):
                oy.append([w,q])

    print(f'Sparsity = {1- (len(oy)/len(cord)**2)}')
    plt.scatter(np.array(oy)[:,0] , np.array(oy)[:,1],s=1)
    jk = 0
    plt.xlim(0-jk,len(cord)+jk)
    plt.ylim(0-jk,len(cord)+jk)
    plt.gca().invert_yaxis()
    plt.show()

#sparser()
plt.show()





