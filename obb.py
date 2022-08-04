import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from statistics import mean
from mpl_toolkits.mplot3d import Axes3D
from statistics import mean
import nibabel as nib


class OBB(object):
    
    def __init__(self, mask, affine):
        '''
            Init of the bounding box class.
        '''
        
        #store affine matrix
        self.affine = affine
        
        #read datapoints from mask
        self.mask = mask
        a,b,c = np.where(mask)
        data = np.vstack([a,b,c])
        
        #transform to world coordinates
        data = nib.affines.apply_affine(self.affine, data.T)
        data = data.T
        #compute centroid
        x,y,z = data
        
        #store final data points
        self.data = data
        #store centroid
        self.centroid = mean(x), mean(y), mean(z)
        #initialize vectors and rectangle coordinates
        self.ovectors = []
        self.rrc = []
    
    def bounding_box(self):
        '''
            Compute the bounding box coordinates using the mask data points.
            :return: 2D array of the box coordinates
        '''
       
        #Based on the following blog post:
        #https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html

        #compute mean
        means = np.mean(self.data, axis=1)
        #compute covariance matrix
        cov = np.cov(self.data)
        #compute eigenvectors and eigenvalues
        eval, evec = LA.eig(cov)
        self.evec = evec
        
        #center data
        centered_data = self.data - means[:,np.newaxis]

        #realing coordinates
        aligned_coords = np.matmul(evec.T, centered_data)
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])
        
        rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                              [y1, y2, y2, y1, y1, y2, y2, y1],
                                                              [z1, z1, z1, z1, z2, z2, z2, z2]])

        realigned_coords = np.matmul(evec, aligned_coords)
        realigned_coords += means[:, np.newaxis]

        rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
        #compute final rotated rectangle coordinates
        rrc += means[:, np.newaxis] 
        self.rrc = rrc
        
        #calculate extent
        #IMPORTANT: In evec dimensions are z-x-y so we need to swap the order
        
        #x-axis
        a = rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2]
        l0 = [(a[0][0] - a[0][1])/2,(a[1][0] - a[1][1])/2, (a[2][0] - a[2][1])/2]
        l0 = LA.norm(l0)
        #y-axis length
        b = rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]]
        l1 = [(b[0][0] - b[0][1])/2,(b[1][0] - b[1][1])/2, (b[2][0] - b[2][1])/2]
        l1 = LA.norm(l1)
        #z-axis
        c = rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3]
        l2 = [(c[0][0] - c[0][1])/2,(c[1][0] - c[1][1])/2,(c[2][0] - c[2][1])/2]
        l2 = LA.norm(l2)
        
        #store the final extent
        self.extent = np.array([l0,l1,l2])
        
        #swap evec columns because of dimensions
        evec[:, [2, 0]] = evec[:, [0, 2]]
        evec[:, [1, 0]] = evec[:, [0, 1]]
        
        #the previously computed eigenvectors are the pre-normalized offset vectors
        self.ovectors = evec

        #generate JSON for each bounding box
        bb = {
            "position": np.array(self.centroid).tolist(),
            "object_oriented_bounding_box": {
                "extent": self.extent.tolist(),
                "orthogonal_offset_vectors": self.ovectors.tolist(),
            }
        }
        
        return bb

 
    def plt_bb(self, show_data=False):
        '''
            Plot the bounding box, eigenvectors and datapoints.
            :param show_data: Flag on whether datapoints will be plotted as well.
            :return: Plot of the bounding box
        '''
        c0,c1,c2 = self.centroid
        evec = self.ovectors*self.extent
        rrc = self.rrc
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if show_data == True:     
            ax.scatter(self.data[0,:], self.data[1,:], self.data[2,:])
        ax.legend()

        # z1 plane boundary
        ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color='r')
        ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color='r')
        ax.plot(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4], color='r')
        ax.plot(rrc[0, [3,0]], rrc[1, [3,0]], rrc[2, [3,0]], color='r')

        # z2 plane boundary
        ax.plot(rrc[0, 4:6], rrc[1, 4:6], rrc[2, 4:6], color='r')
        ax.plot(rrc[0, 5:7], rrc[1, 5:7], rrc[2, 5:7], color='r')
        ax.plot(rrc[0, 6:], rrc[1, 6:], rrc[2, 6:], color='r')
        ax.plot(rrc[0, [7, 4]], rrc[1, [7, 4]], rrc[2, [7, 4]], color='r')

        # z1 and z2 connecting boundaries
        ax.plot(rrc[0, [0, 4]], rrc[1, [0, 4]], rrc[2, [0, 4]], color='r')
        ax.plot(rrc[0, [1, 5]], rrc[1, [1, 5]], rrc[2, [1, 5]], color='r')
        ax.plot(rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]], color='r')
        ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]],color='r' )

        #eigenbasis
        ax.plot([c0, c0 + evec[0, 0]],  [c1, c1+ evec[1, 0]], [c2, c2 + evec[2, 0]], color='b', linewidth=4)
        ax.plot([c0, c0+ evec[0, 1]],  [c1, c1+evec[1, 1]], [c2, c2+evec[2, 1]], color='g', linewidth=4)
        ax.plot([c0,c0+ evec[0, 2]],  [c1, c1+evec[1, 2]], [c2, c2+evec[2, 2]], color='y', linewidth=4)
        
        
        plt.show()
    
        
    
 