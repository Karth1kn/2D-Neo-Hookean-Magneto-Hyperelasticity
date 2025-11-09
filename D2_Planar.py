import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import csv
import os
import time
from scipy.linalg import solveh_banded
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix 
from scipy.sparse.linalg import cg
E = 2e9 #N/m**2
y = 0.25
t = 10e-3
cord = np.array([[0.0,0.0] , [1.0,0.0] , [1.0,0.75] , [0.0,0.75]]) #m
con = np.array([[0,2,3] , [0,1,2]])
force = [[2 , [75000,0]]]
bc_lst = np.array([0,1 , 3 , 6,7])


np.set_printoptions(precision=4)

def timer(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        mat = func(*args, **kwargs)
        t2 = time.time()
        print(f'{func.__name__} finished in: {t2-t1}')
        return mat
    return wrapper

script_dir = os.path.dirname(os.path.abspath(__file__))
coord_path = os.path.join(script_dir, 'Plane_coord.csv')
con_path = os.path.join(script_dir, 'Plane_connect.csv')
prop_path = os.path.join(script_dir, 'Plane_properties.csv')
bc_path = os.path.join(script_dir, 'Plane_bound_con.csv')
force_path = os.path.join(script_dir, 'Plane_force_list.csv')
edge_path = os.path.join(script_dir, 'Plane_edges.csv')
with open(prop_path,'r')as fl:
    props = []
    for i in csv.reader(fl):
        for j in i:
            props.append(float(j))
    E = props[0]
    y = props[1]
    Yield = props[2]
    t =props[3]

with open(coord_path, 'r') as file:
    try:
        val_list = []
        for i in csv.reader(file):
            val_list.append([float(j) for j in i])
        val_array= np.array(val_list)
        with open(con_path,'r') as con:
            con_list= []

            for c in csv.reader(con):
                con_list.append([int(b) for b in c])
            con_array= np.array(con_list)

    except Exception:
        print('error')


with open(edge_path,'r') as edg:
    edge_list= []

    for c in csv.reader(edg):
        edge_list.append([int(b) for b in c])
    edge_array= np.array(edge_list)


with open(bc_path, 'r') as bc:
        bc_lst = []
        c = 0
        for c1 in csv.reader(bc):
            pos = int(c1[0])*2
            if c1[1]=='x':
                if c1[0]==c:
                    continue
                bc_lst.append(pos)
            elif c1[1] == 'y':
                if c1[0]==c:
                    continue
                bc_lst.append(pos+1)

""" with open(force_path, 'r') as force:
        force_list =[]
        lst = csv.reader(force)
        for row in lst:
            if row == []: continue
            for i in row[2:]:
                force_list.append([int(i),[float(j) for j in row[:2]]]) """

with open(force_path, 'r') as force:
    force_list = {}
    lst = csv.reader(force)
    
    for row in lst:
        if row == []: 
            continue
        for i in row[2:]:
            force_list[int(i)] = [float(j) for j in row[:2]]

#print(force_list)
""" with open(force_path, 'r') as force:
        force_list =[]
        lst = csv.reader(force) 
        for row in lst:
            if row == []: continue
            for i in range(0, len(row[2:]), 2):
                force_list.append([[int(row[i+2]), int(row[i+3])],[float(j) for j in row[:2]]]) """
#print(np.array(force_list)[:])

#E, y, t =  1, 0.2, 1
#E, y, t =  30e6, 0.25, 1
D = (E/(1-y**2))*np.array([ [1, y,       0],
                            [y, 1,       0],
                            [0, 0, (1-y)/2]])


Ds = (E/((1+y)*(1-2*y)))*np.array([ [1-y,   y,         0],
                                    [  y, 1-y,         0],
                                    [  0,   0, (1-2*y)/2]])

def force_matrix(node,force,coord):
    f_matrix= np.zeros([2*len(coord),1])
    k = 2*node
    for i in range(2):
        f_matrix[k+i] += force[i]
    return f_matrix

@timer
def cst():
    fig, ax = plt.subplots()

    cord=val_array
    con = con_array
    force_mat=force_matrix(0,[0.0,0.0], cord)

    for ac in force_list:
        force_mat += force_matrix(*ac, cord)


    def area(element):
        area = 0.5*np.linalg.det(np.array([[1, *cord[element[0]]],
                                            [1, *cord[element[1]]],
                                            [1, *cord[element[2]]]]))
        return abs(area)

    def local_stiffness_matrix(e):
        a = area(e)
        x1, x2, x3 = cord[e[0]][0], cord[e[1]][0], cord[e[2]][0]
        y1, y2, y3 = cord[e[0]][1], cord[e[1]][1], cord[e[2]][1]
        q1, q2, q3, r1, r2, r3 = y2-y3, y3-y1, y1-y2, x3-x2, x1-x3, x2-x1

        B = (0.5/a)*np.array([[q1,  0, q2,  0, q3,  0],
                                [ 0, r1,  0, r2,  0, r3],
                                [r1, q1, r2, q2, r3, q3]])
        
        #print(B)
        BtD = np.transpose(B)@D
        return (a*t)*BtD@B, B

    #print(local_stiffness_matrix(con[1]))

    def global_stiff_mat(coord_list):
        matrix= np.zeros([len(cord)*2,len(cord)*2])
        for p,i in enumerate(coord_list):
            t1 = np.array([[i[0],i[0]], [i[0],i[1]], [i[0],i[2]], [i[1],i[0]], [i[1],i[1]], [i[1],i[2]], [i[2],i[0]], [i[2],i[1]], [i[2],i[2]]])
            t2= np.array([[0,0], [0,2], [0,4], [2,0], [2,2], [2,4], [4,0], [4,2], [4,4]])
            k= local_stiffness_matrix(i)[0]
            for j in range(9):
                matrix[2*t1[j,0]:2*t1[j,0]+2 , 2*t1[j,1]:2*t1[j,1]+2] += k[t2[j,0]:t2[j,0]+2 , t2[j,1]:t2[j,1]+2]
        return matrix
    

    def boundary_conditions(matrix,bc_list):
        matrix= np.delete(matrix,bc_list,0)
        if len(matrix[0])!=1:
            matrix= np.delete(matrix,bc_list,1)
        return matrix


    global_stiff_matrix= global_stiff_mat(con)
    #print(global_stiff_matrix)
    #APPLYING BOUNDARY CONDITIONS
    bc_stiff_mat= boundary_conditions(global_stiff_matrix,bc_lst)
    #print(bc_stiff_mat)
    bc_force_matrix= boundary_conditions(force_mat,bc_lst)
    #print(bc_force_matrix)

    bc_stiff_mat_inv= np.linalg.inv(bc_stiff_mat) #matrix_inverter(bc_stiff_mat)

    #MULTIPLYING INVERSE TO FORCE MATRIX
    nodal_disp= bc_stiff_mat_inv@bc_force_matrix

    for i in bc_lst:
        nodal_disp= np.insert(nodal_disp,i,0)
    nodal_disp1 = nodal_disp.copy()
    shape_nodal_disp= nodal_disp1.reshape(len(cord),2)

    cord += shape_nodal_disp


    def stress(ele):
        B = local_stiffness_matrix(ele)[1]
        U = np.array([nodal_disp[ele[i]*2 : ele[i]*2+2] for i in range(3) ]).ravel()
        Bu = B@U
        stress = D@Bu
        return stress

    for i in con:
        pass
        #print(stress(i))


    #print(cord)
    for j in (con):
        triangle = np.array([cord[j[0]], cord[j[1]], cord[j[2]]])

        
        fos= stress(j)[0]/(Yield*100)
        if fos**2>=1:
            r,g,b,a = 0 ,1 ,0 ,1
            plt.annotate('material will fail at this load',(3000,-5000),color= 'white')
            ax.fill(triangle[:,0], triangle[:,1], color=(r,g,b,a) )

        elif fos<=0:
            r,g,b,a = 1+fos, 1+fos, 1,  1
            ax.fill(triangle[:,0], triangle[:,1], color=(r,g,b,a) )
        elif fos>0:
            r,g,b,a= 1, 1-fos, 1-fos,  1         
            ax.fill(triangle[:,0], triangle[:,1], color=(r,g,b,a) )

        #ax.fill(triangle[:,0], triangle[:,1], color = 'blue' )



    plt.axis('equal')
    ax.set_facecolor('black')



@timer
def isoparametric():
    thick = t
    cord=val_array
    con = con_array
    force_lis = np.array(force_list)
    fig, ax = plt.subplots()

    #cord = np.array([[0,0], [3,2], [4,4], [2,5]])
    #cord = np.array([[0,0], [1,0], [1,1], [0,1]])
    #cord = np.array([[3.0,2.0], [5.0,2.0], [5.0,4.0], [3.0,4.0]])
    #cord = np.array([[0.0,0], [8,0], [5,4], [0,4]])
    #cord = np.array([[0,0],[1,0.1],[1.2,1.2],[0.2,1]])
    #con = np.array([[0,1,2,3]])
    #force_lis = np.array([[[1,2], [2000,0]]])
    #thick = 0.1

    def forces(element, force, n, f):   
        m = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
        ln = lambda n1,n2: (( ( cord[(n2),0] - cord[(n1),0] )**2 + ( cord[(n2),1] - cord[(n1),1] )**2)**0.5)
        l = ln(int(element[0]), int(element[1]))
        #print(l)
        def shapefn(s,t):
            shape = lambda i: 0.25*(1 + m[i,0]*s)*(1 + m[i,1]*t)
            arr = thick*0.5*l*np.array([[force[0]*shape(n[0])],
                                [force[1]*shape(n[0])],
                                [force[0]*shape(n[1])],
                                [force[1]*shape(n[1])]])
            return arr
        if f==0:
            return 2*shapefn(0,-1)
        elif f==1:
            return 2*shapefn(1,0)
        elif f==2:
            return 2*shapefn(0,1) 
        else:
            return 2*shapefn(-1,0)

    def surface_load_matrix():
        force_mat=force_matrix(0,[0.0,0.0], cord)
        rotate = lambda a,b: ([i for i in range(a)][b:] + [i for i in range(a)][:b])[:2]

        for force in force_lis:
            for j in range(4):
                if np.any(np.all(con[:, rotate(4,j)] == force[0], axis =1)): #force[0] in con[:, rotate(4,j)]:
                    fo = forces(*force, rotate(4,j), j)
                    force_mat += force_matrix(int(force[0,0]), fo[:2][:,0], cord)
                    force_mat += force_matrix(int(force[0,1]), fo[2:][:,0], cord)
                    break
        return force_mat
    force_mat = surface_load_matrix()
    #print(force_mat)



    def Bmat(e, n, c):
        x1, x2, x3, x4 = cord[c[0]][0], cord[c[1]][0], cord[c[2]][0], cord[c[3]][0]
        y1, y2, y3, y4 = cord[c[0]][1], cord[c[1]][1], cord[c[2]][1], cord[c[3]][1]

        j11 = 0.25*(-(1-n)*x1 + (1-n)*x2 + (1+n)*x3 - (1+n)*x4)
        j12 = 0.25*(-(1-n)*y1 + (1-n)*y2 + (1+n)*y3 - (1+n)*y4)
        j21 = 0.25*(-(1-e)*x1 - (1+e)*x2 + (1+e)*x3 + (1-e)*x4)
        j22 = 0.25*(-(1-e)*y1 - (1+e)*y2 + (1+e)*y3 + (1-e)*y4)
        det = j11*j22 - j12*j21

        B1 = np.array([[ j22, -j12,    0,   0],
                    [   0,    0, -j21, j11],
                    [-j21,  j11,  j22,-j12]])

        B2 = np.array([[-(1-n), 0, (1-n), 0, (1+n), 0, -(1+n), 0],
                       [-(1-e), 0, -(1+e), 0, (1+e), 0, (1-e), 0],
                       [0, -(1-n), 0, (1-n), 0, (1+n), 0, -(1+n)],
                       [0, -(1-e), 0, -(1+e), 0, (1+e), 0, (1-e)]])

        B = B1@B2/(4*det)

        return B,det
    
    def local_stiffness_matrix(c):
        def integ(e,n):
            B = Bmat(e,n,c)[0]
            det = Bmat(e,n,c)[1]
            BD = np.transpose(B)@D
            K = BD@B
            return det*thick*K
        
        r3 = 1/math.sqrt(3)
        Kf = integ(r3, r3) + integ(r3, -r3) + integ(-r3, r3) + integ(-r3, -r3) # FULL INTEGRATION
        #Kf = 4*integ(0,0) #REDUCED INTEGRATION
        return Kf
    

    #print(Bmat(0.57735,-0.57735,con[0]))
    #print(local_stiffness_matrix(con[0]))

    @timer
    def global_stiff_mat(coord_list):
        matrix= np.zeros([len(cord)*2,len(cord)*2])
        for p,i in enumerate(coord_list):
            t1 = np.array([[i[x], i[y]] for x in range(4) for y in range(4)])
            t2 = np.array([[r,s]for r in range(0,7,2) for s in range(0,7,2)])
            k= local_stiffness_matrix(i)
            for j in range(16):
                matrix[2*t1[j,0]:2*t1[j,0]+2 , 2*t1[j,1]:2*t1[j,1]+2] += k[t2[j,0]:t2[j,0]+2 , t2[j,1]:t2[j,1]+2]
        return matrix
    
    
    def boundary_conditions(matrix,bc_list):
        matrix= np.delete(matrix,bc_list,0)
        if len(matrix[0])!=1:
            matrix= np.delete(matrix,bc_list,1)
        return matrix



    @timer
    def main_sequence(cord):
        global global_stiff_matrix
        global_stiff_matrix= global_stiff_mat(con)

        bc_stiff_mat= boundary_conditions(global_stiff_matrix,bc_lst)

        bc_force_matrix= boundary_conditions(force_mat,bc_lst)

        ta = time.time()
        #bandwidth = max([max(i)-min(i) for i in con ])*2+2
        #print(bandwidth)
        #diagonals = np.array([np.append(np.diag(bc_stiff_mat, -i), np.zeros([1,i])[0]) for i in range(bandwidth)  ])
        global nodal_disp
        #nodal_disp = solveh_banded(diagonals, bc_force_matrix, lower= True)
        tb = time.time()


        tc = time.time()
        nodal_disp= np.linalg.inv(bc_stiff_mat)@bc_force_matrix
        #A = csc_matrix(bc_stiff_mat)
        #B = bc_force_matrix
        #nodal_disp, xx = cg(A, B, atol=0.1)
        td = time.time()


        print('solution time OPTIMISED: ', tb-ta)
        print('solution time REGULAR:', td-tc)
        for i in bc_lst:
            nodal_disp= np.insert(nodal_disp,i,0)
        nodal_disp1 = nodal_disp.copy()
        shape_nodal_disp= nodal_disp1.reshape(len(cord),2)

        

        return shape_nodal_disp

    cord += main_sequence(cord)


    @timer
    def sparser():
        fig1, ax1 = plt.subplots()
        oy = set()
        sparse_matrix = csr_matrix(global_stiff_matrix)
        rows, cols = sparse_matrix.nonzero()
        bandwidth = max([max(i)-min(i) for i in con ])*2+2
        loc = np.argmax([max(i)-min(i) for i in con ])

        rn = 2000
        plt.plot([max(con[loc])-rn, max(con[loc]) +rn], [min(con[loc])-rn, min(con[loc])+rn], color = 'red')

        for idx in range(len(rows)):
            q = rows[idx] // 2
            w = cols[idx] // 2
            if q < len(cord) and w < len(cord):
                #oy.add((w, q))
                plt.scatter(w,q, s=1)

        oy = list(oy)

        #print(f'Sparsity = {1- (len(oy)/len(cord)**2)}')
        #plt.scatter(np.array(oy)[:,0] , np.array(oy)[:,1],s=1)

        plt.gca().invert_yaxis()
        plt.axis('equal')

    # sparser()


    def stress(ele):
        U = np.array([nodal_disp[ele[i]*2 : ele[i]*2+2] for i in range(4) ]).ravel()
        def integ(e,n):
            B = Bmat(e,n,ele)[0]
            DB = D@B
            S = DB@U
            return S

        r3 = 1/math.sqrt(3)
        Ss = integ(r3, r3) + integ(r3, -r3) + integ(-r3, r3) + integ(-r3, -r3)
        #Kf = 4*integ(0,0)
        return Ss




    @timer
    def plotter():
        for j in (con):
            quadr = np.array([cord[j[0]], cord[j[1]], cord[j[2]], cord[j[3]]])

            fos= stress(j)[0]/(Yield*10e6)
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

            #ax.fill(quadr[:,0], quadr[:,1], color = 'blue' )

    plotter()

    plt.axis('equal')
    ax.set_facecolor('black')
    
#cst()
#isoparametric()




#plt.show()
