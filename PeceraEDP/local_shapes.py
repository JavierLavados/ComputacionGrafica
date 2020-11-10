""" Local shapes module, containing the logic for creating shapes"""

import numpy as np

import basic_shapes as bs


##################### Triangulos ####################################3

def createColorTriangleIndexation(start_index, a, b, c, color):
    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors             
        a[0], a[1], a[2], color[0], color[1], color[2],
        b[0], b[1], b[2], color[0], color[1], color[2],
        c[0], c[1], c[2], color[0], color[1], color[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index, start_index+1, start_index+2,
         start_index+2, start_index+3, start_index
        ]

    return (vertices, indices)


def createColorNormalsTriangleIndexation(start_index, a, b, c, color):
    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                        normals
        a[0], a[1], a[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2], color[0], color[1], color[2], v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index, start_index+1, start_index+2,
         start_index+2, start_index+3, start_index
        ]

    return (vertices, indices)



def createTextureNormalsTriangleIndexation(start_index, a, b, c):
    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                        normals
        a[0], a[1], a[2], 0,0, v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2], 1,0, v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2], 0,1, v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index, start_index+1, start_index+2,
         start_index+2, start_index+3, start_index
        ]

    return (vertices, indices)

######################## Cuadrados #############################################

def createColorQuadIndexation(start_index, a, b, c, d, color):
    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors
        a[0], a[1], a[2], color[0], color[1], color[2],
        b[0], b[1], b[2], color[0], color[1], color[2],
        c[0], c[1], c[2], color[0], color[1], color[2],
        d[0], d[1], d[2], color[0], color[1], color[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index, start_index+1, start_index+2,
         start_index+2, start_index+3, start_index
        ]

    return (vertices, indices)




def createColorNormalsQuadIndexation(start_index, a, b, c, d, color):

    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                 normals
        a[0], a[1], a[2], color[0], color[1], color[2],  v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2], color[0], color[1], color[2],  v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2], color[0], color[1], color[2],  v1xv2[0], v1xv2[1], v1xv2[2],
        d[0], d[1], d[2], color[0], color[1], color[2],  v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index, start_index+1, start_index+2,
         start_index+2, start_index+3, start_index
        ]
    
    return (vertices, indices)

def createTextureNormalsQuadIndexation(start_index, a, b, c, d):

    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                 normals
        a[0], a[1], a[2],    0,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2],    1,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2],    1,0,     v1xv2[0], v1xv2[1], v1xv2[2],
        d[0], d[1], d[2],    0,0,     v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index, start_index+1, start_index+2,
         start_index+2, start_index+3, start_index
        ]
    
    return (vertices, indices)



######################### Para la pista ###############################

def createTextureNormalsQuadIndexationFirst(start_index, a, b, c, d):

    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                 normals
        a[0], a[1], a[2],    1,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2],    0,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2],    1,1,     v1xv2[0], v1xv2[1], v1xv2[2],
        d[0], d[1], d[2],    0,1,     v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index+1, start_index, start_index+2,
         start_index+2, start_index+3, start_index+1
        ]
    
    return (vertices, indices)

def createTextureNormalsQuadIndexationSecond(start_index, a, b, c, d):

    # Computing normal from a b c
    v1 = np.array(a-b)
    v2 = np.array(b-c)
    v1xv2 = np.cross(v1, v2)

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #        positions               colors                 normals
        a[0], a[1], a[2],    1,0,     v1xv2[0], v1xv2[1], v1xv2[2],
        b[0], b[1], b[2],    0,0,     v1xv2[0], v1xv2[1], v1xv2[2],
        c[0], c[1], c[2],    1,0,     v1xv2[0], v1xv2[1], v1xv2[2],
        d[0], d[1], d[2],    0,0,     v1xv2[0], v1xv2[1], v1xv2[2]
    ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         start_index+1, start_index, start_index+2,
         start_index+2, start_index+3, start_index+1
        ]
    
    return (vertices, indices)

#############################Figuras##################################################################333
def generateCylinder(latitudes, image_filename, R = 1.0, z_top=1.0, z_bottom=0.0):

    vertices = []
    indices = []

    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0

    # We generate a rectangle for every latitude, 
    for _ in range(latitudes):
        # d === c
        # |     |
        # |     |
        # a === b

        a = np.array([R*np.cos(theta), R*np.sin(theta), z_bottom])
        b = np.array([R*np.cos(theta + dtheta), R*np.sin(theta + dtheta), z_bottom])
        c = np.array([R*np.cos(theta + dtheta), R*np.sin(theta + dtheta), z_top])
        d = np.array([R*np.cos(theta), R*np.sin(theta), z_top])

        theta = theta + dtheta

        _vertex, _indices = createTextureNormalsQuadIndexation(start_index, a, b, c, d)

        vertices += _vertex
        indices  += _indices
        start_index += 4

    # add top cover
   
    theta = 0
    dtheta = 2 * np.pi / 15

    for _ in range(15):
        # Top
        a = np.array([0, 0, z_top])
        b = np.array([R * np.cos(theta), R * np.sin(theta), z_top])
        c = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_top])

        _vertex, _indices = createTextureNormalsTriangleIndexation(start_index, a, b, c)

        vertices += _vertex
        indices  += _indices
        start_index += 3
        theta += dtheta
    for _ in range(15):
        # Botton
        a = np.array([0, 0, z_bottom])
        b = np.array([R * np.cos(theta), R * np.sin(theta), z_bottom])
        c = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_bottom])

        _vertex, _indices = createTextureNormalsTriangleIndexation(start_index, a, b, c)

        vertices += _vertex
        indices  += _indices
        start_index += 3
        theta += dtheta


    return bs.Shape(vertices,indices,image_filename)


def generateCylinderH(latitudes, image_filename, R = 1.0, z_top=1.0, z_bottom=0.0):

    vertices = []
    indices = []

    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0

    # We generate a rectangle for every latitude, 
    for _ in range(latitudes):
        # d === c
        # |     |
        # |     |
        # a === b

        a = np.array([R*np.cos(theta), R*np.sin(theta), z_bottom])
        b = np.array([R*np.cos(theta + dtheta), R*np.sin(theta + dtheta), z_bottom])
        c = np.array([R*np.cos(theta + dtheta), R*np.sin(theta + dtheta), z_top])
        d = np.array([R*np.cos(theta), R*np.sin(theta), z_top])

        theta = theta + dtheta

        _vertex, _indices = createTextureNormalsQuadIndexation(start_index, a, b, c, d)

        vertices += _vertex
        indices  += _indices
        start_index += 4

    return bs.Shape(vertices,indices,image_filename)

def generateCirculo(latitudes, R = 1.0):

    vertices = []
    indices = []

    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0

    for _ in range(latitudes):
        # Botton
        a = np.array([0, 0, 0])
        b = np.array([R * np.cos(theta), R * np.sin(theta),0])
        c = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), 0])

        _vertex, _indices = createColorTriangleIndexation(start_index, a, b, c,[0,0,0])

        vertices += _vertex
        indices  += _indices
        start_index += 3
        theta += dtheta


    return bs.Shape(vertices,indices)

def generatecono(latitudes, color, R = 1.0, z_top=1.0, z_bottom=0.0):

    vertices = []
    indices = []

    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0

    # We generate a rectangle for every latitude, 
    for _ in range(latitudes):
        # d === c
        # |     |
        # |     |
        # a === b

        c = np.array([R*np.cos(theta + dtheta), R*np.sin(theta + dtheta), z_top])
        d = np.array([R*np.cos(theta), R*np.sin(theta), z_top])
        a = np.array([0,0,z_bottom])

        theta = theta + dtheta

        _vertex, _indices = createColorNormalsTriangleIndexation(start_index, a, c, d, color)

        vertices += _vertex
        indices  += _indices
        start_index += 4

    # add top cover
   
    theta = 0
    dtheta = 2 * np.pi / 15

    for _ in range(15):
        # Top
        a = np.array([0, 0, z_top])
        b = np.array([R * np.cos(theta), R * np.sin(theta), z_top])
        c = np.array([R * np.cos(theta + dtheta), R * np.sin(theta + dtheta), z_top])

        _vertex, _indices = createColorNormalsTriangleIndexation(start_index, a, b, c,color)

        vertices += _vertex
        indices  += _indices
        start_index += 3
        theta += dtheta
    return bs.Shape(vertices,indices)

def generateCylinderColor(latitudes, R ,z_top,z_bottom,color):
    vertices = []
    indices = []

    # Angle step
    dtheta = 2 * np.pi / latitudes
    theta = 0
    start_index = 0

    # We generate a rectangle for every latitude, 
    for _ in range(latitudes):
        # d === c
        # |     |
        # |     |
        # a === b

        a = np.array([R*np.cos(theta), R*np.sin(theta), z_bottom])
        b = np.array([R*np.cos(theta + dtheta), R*np.sin(theta + dtheta), z_bottom])
        c = np.array([R*np.cos(theta + dtheta), R*np.sin(theta + dtheta), z_top])
        d = np.array([R*np.cos(theta), R*np.sin(theta), z_top])

        theta = theta + dtheta

        _vertex, _indices = createColorQuadIndexation(start_index, a, b, c, d,color)

        vertices += _vertex
        indices  += _indices
        start_index += 4

    return bs.Shape(vertices,indices)


def esfera(nTheta, nPhi,color):
    vertices = []
    indices = []

    theta_angs = np.linspace(0, np.pi, nTheta, endpoint=True)
    phi_angs = np.linspace(0, 2 * np.pi, nPhi, endpoint=True)

    start_index = 0

    for theta_ind in range(len(theta_angs)-1): # vertical
        cos_theta = np.cos(theta_angs[theta_ind]) # z_top
        cos_theta_next = np.cos(theta_angs[theta_ind + 1]) # z_bottom

        sin_theta = np.sin(theta_angs[theta_ind])
        sin_theta_next = np.sin(theta_angs[theta_ind + 1])

        # d === c <---- z_top
        # |     |
        # |     |
        # a === b  <--- z_bottom
        # ^     ^
        # phi   phi + dphi
        for phi_ind in range(len(phi_angs)-1): # horizontal
            cos_phi = np.cos(phi_angs[phi_ind])
            cos_phi_next = np.cos(phi_angs[phi_ind + 1])
            sin_phi = np.sin(phi_angs[phi_ind])
            sin_phi_next = np.sin(phi_angs[phi_ind + 1])
            # we will asume radius = 1, so scaling should be enough.
            # x = cosφ sinθ
            # y = sinφ sinθ
            # z = cosθ

            #                     X                             Y                          Z
            a = np.array([cos_phi      * sin_theta_next, sin_phi * sin_theta_next     , cos_theta_next])
            b = np.array([cos_phi_next * sin_theta_next, sin_phi_next * sin_theta_next, cos_theta_next])
            c = np.array([cos_phi_next * sin_theta     , sin_phi_next * sin_theta     , cos_theta])
            d = np.array([cos_phi * sin_theta          , sin_phi * sin_theta          , cos_theta])

            _vertex, _indices = createColorNormalsQuadIndexation(start_index, a, b, c, d, color)

            vertices += _vertex
            indices  += _indices
            start_index += 4

    return bs.Shape(vertices, indices)