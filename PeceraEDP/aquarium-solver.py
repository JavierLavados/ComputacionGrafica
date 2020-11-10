# coding=utf-8

import numpy as np
import sys
import json
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

archivo = sys.argv[1]

# Obtengo la informacion del archivo json
Data=None
with open(archivo) as file:
    data = json.load(file)
    Data=data   

W=Data["width"]  # Ancho
L=Data["lenght"] # Largo
H=Data["height"] # Altura

F=Data["window_loss"] # Condicion de neuman de las paredes
h = 0.2 # Paso 

heater_a=Data["heater_a"] # Calentador 1
heater_b=Data["heater_b"] # Calentador 2
T=Data["ambient_temperature"] # Temperatura ambiental

# Number of unknowns
nx = int(W / h) + 1 # Discretizacion en X
ny = int(L / h) + 1 # Discretizacion en Y
nk = int(H / h)     # Discretizacion en Z

# In this case, the domain is an aquarium with parallelepiped form
N = nx * ny * nk

# Funcion que entrega la discretizacion del punto i,j,k a un g
def getG(i,j,k):
    return  i+j * nx +k*nx*ny

# Obtiene las cordenadas i,j,k desde g
def getIJK(g):
    i = (g %( nx*ny))%nx
    j = (g %( nx*ny))//nx
    k = g // (nx*ny)
    return (i, j, k)



A = sparse.lil_matrix((N,N)) # Usamos una matriz sparse para ahorrar memoria

# In this vector we will write all the right side of the equations
b = np.zeros((N,))

# Note: To write an equation is equivalent to write a row in the matrix system

# We iterate over each point inside the domain
# Each point has an equation associated
# The equation is different depending on the point location inside the domain

for k in range(0, nk):
    for j in range(0, ny):
        for i in range(0, nx):
            # We will write the equation associated with row K
            g = getG(i, j, k)
            # We obtain indices of the other coefficients
            g_right = getG(i+1, j, k)
            g_left = getG(i-1, j, k)
            g_front = getG(i, j+1, k)
            g_back = getG(i, j-1, k)
            g_up = getG(i, j, k+1)
            g_down = getG(i, j, k-1)
            
            # Depending on the location of the point, the equation is different

            # Interior
            if (1 <= i) and (i <= nx - 2) and (1 <= j) and (j <= ny - 2) and (1 <= k) and (k <= nk-2):
                
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = 0

            ### Caras ###
     
            # Cara derecha
            elif i == nx-1 and (1 <= j) and (j <= ny - 2) and (1 <= k) and (k <= nk-2):
                A[g, g_left] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara izquierda
            elif i == 0 and (1 <= j) and (j <= ny - 2) and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F
            
            # Cara Frontal
            elif (1 <= i) and (i <= nx-2) and j == ny-1 and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_back] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara Trasera
            elif (1 <= i) and (i <= nx-2) and j == 0 and (1 <= k) and (k <= nk-2):
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F
    
            # Cara superior
            elif (1 <= i) and (i <= nx - 2) and (1 <= j) and (j <= ny-2) and k == nk-1:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -T

            # Calentador A
            elif (nx//3 <= i) and (i <= 2*nx//3) and (ny-(2*ny//5) <= j) and (j <= ny-(ny//5)) and k == 0:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g] = -6
                b[g] =  -heater_a

            # Calentador B
            elif (nx//3 <= i) and (i <= 2*nx//3) and (ny//5 <= j) and (j <= 2*ny//5) and k == 0:
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 1
                A[g, g] = -6
                b[g] =  -heater_b

            # Cara Inferior
            elif (1 <= i) and (i <= nx - 2) and (1 <= j) and (j <= ny-2) and k == 0:
                
                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = 0


            ##### Esquinas

            # right front up
            elif i == nx-1 and j == ny-1 and k == nk-1:

                A[g, g_left] = 2
                A[g, g_back] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T
            
            # left front up
            elif i == 0 and j == ny-1 and k == nk-1:

                A[g, g_right] = 2
                A[g, g_back] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T
            
            # right back up
            elif i == nx-1 and j == 0 and k == nk-1:

                A[g, g_left] = 2
                A[g, g_front] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T
            
            # left back up
            elif i == 0 and j == 0 and k == nk-1:

                A[g, g_right] = 2
                A[g, g_front] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F -T

            # front right down
            elif i==nx-1 and  j== ny-1 and k==0:

                A[g, g_left] = 2
                A[g, g_back] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F

            #  front left down
            elif i==0 and j==ny-1 and k==0:

                A[g, g_right] = 2
                A[g, g_back] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F

            # back right down
            elif i==nx-1 and j==0 and k==0 :

                A[g, g_left] = 2
                A[g, g_front] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F

            # back left down
            elif i==0 and  j==0 and k==0:

                A[g, g_right] = 2
                A[g, g_front] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -4 * h * F



            ### Aristas ###

            # Cara derecha , abajo
            elif i == nx-1 and (1 <= j) and (j <= ny - 2) and k == 0:

                A[g, g_left] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara izquierda , abajo
            elif i == 0 and (1 <= j) and (j <= ny - 2) and k == 0:

                A[g, g_right] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara frontal , abajo
            elif (1 <= i) and (i <= nx-2) and j == ny-1 and k == 0:

                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_back] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara trasera , abajo
            elif (1 <= i) and (i <= nx-2) and j == 0 and k == 0:

                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 2
                A[g, g_up] = 2
                A[g, g] = -6
                b[g] = -2 * h * F

            # Cara frontal , derecha
            elif i == nx-1 and j == ny-1 and (1 <= k) and (k <= nk-2):

                A[g, g_left] = 2
                A[g, g_back] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara frontal , izquierda
            elif i == 0 and j == ny-1 and (1 <= k) and (k <= nk-2):

                A[g, g_right] = 2
                A[g, g_back] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara trasera , derecha
            elif i == nx-1 and j == 0 and (1 <= k) and (k <= nk-2):

                A[g, g_left] = 2
                A[g, g_front] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara trasera , izquierda
            elif i == 0 and j == 0 and (1 <= k) and (k <= nk-2):

                A[g, g_right] = 2
                A[g, g_front] = 2
                A[g, g_up] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -4 * h * F
            
            # Cara superior , derecha
            elif i == nx-1 and (1 <= j) and (j <= ny - 2) and k == nk-1:

                A[g, g_left] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T

            # Cara superior , izquierda
            elif i == 0 and (1 <= j) and (j <= ny - 2) and k == nk-1:

                A[g, g_right] = 2
                A[g, g_front] = 1
                A[g, g_back] = 1
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T
            
            # Cara superior , arriba
            elif (1 <= i) and (i <= nx-2) and j == ny-1 and k == nk-1:

                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_back] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T

            # Cara superior , abajo
            elif (1 <= i) and (i <= nx-2) and j == 0 and k == nk-1:

                A[g, g_right] = 1
                A[g, g_left] = 1
                A[g, g_front] = 2
                A[g, g_down] = 1
                A[g, g] = -6
                b[g] = -2 * h * F -T
                
            else:
                print("Point (" + str(i) + ", " + str(j) + ", " + str(k) + ") missed!")
                print("Associated point index is " + str(K))
                raise Exception()



# Solving our system
x = linalg.spsolve(A, b)

# Now we return our solution to the 3d discrete domain
# In this matrix we will store the solution in the 3d domain
u = np.zeros((nx,ny,nk))

for g in range(0, N):
    i,j,k = getIJK(g)
    u[i,j,k] = x[g]

# Adding the borders, as they have known values
ub = np.zeros((nx,ny,nk+1))
ub[0:nx, 0:ny, 0:nk] = u[:,:,:]

# Dirichlet boundary condition on the top side
ub[0:nx, 0:ny, nk] = T # agrego la temperatura ambiental arriba


name=Data["filename"] # Guardo el archivo

np.save(name,ub)

Y,X,Z=np.meshgrid(np.linspace(0,L,ny),np.linspace(0,W,nx),np.linspace(0,H,nk+1))

fig = plt.figure()
ax = fig.gca(projection='3d')

scat = ax.scatter(X,Y,Z, c=ub, alpha=0.5, s=100, marker='s')

fig.colorbar(scat, shrink=0.5, aspect=5) # This is the colorbar at the side

# Showing the result
ax.set_title('Laplace equation solution from aquarium')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()
