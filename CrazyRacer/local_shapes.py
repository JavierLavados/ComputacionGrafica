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

def generateCirculo(latitudes, texture, R = 1.0):

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

        _vertex, _indices = createTextureNormalsTriangleIndexation(start_index, a, b, c)

        vertices += _vertex
        indices  += _indices
        start_index += 3
        theta += dtheta


    return bs.Shape(vertices,indices,texture)

def generatecono(latitudes, image_filename, R = 1.0, z_top=1.0, z_bottom=0.0):

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

        _vertex, _indices = createTextureNormalsTriangleIndexation(start_index, a, c, d)

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
    return bs.Shape(vertices,indices,image_filename)

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