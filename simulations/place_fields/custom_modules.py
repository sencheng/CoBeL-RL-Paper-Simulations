from cobel.frontends.frontends_blender import FrontendBlenderInterface
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.spatial_representations.topology_graphs.simple_topology_graph  import HexagonalGraph
import gym
import random
import numpy as np
from PyQt5.QtCore import QRectF
import cv2


def add_gaussian_noise(observation, mean, var) :    
    row, col, ch= observation.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    return observation + gauss
    
def crop_image(image, image_dim, view_angle) : 
    keep_pixels = image_dim[0] * view_angle / 360.0
    remove_pixels = image_dim[0] - keep_pixels
    rm = np.ceil(remove_pixels/2).astype('int32')
    cropped_image = image[:,rm:rm+np.ceil(keep_pixels).astype('int32')]
    return cropped_image
    

class ImageObservationFOV(ImageObservationBaseline) : 
    
    def __init__(self, world, guiParent, with_GUI=True, image_dims=(30, 1),
                 view_angle=360.0, noise=(0.0,0.0)) :
        super().__init__(world, guiParent, with_GUI=with_GUI, image_dims=image_dims)
        self.view_angle = view_angle 
        self.observation = crop_image(image=np.zeros((self.image_dims[1],
                                                      self.image_dims[0],3)),
                                         image_dim=image_dims, 
                                         view_angle=self.view_angle)
        self.noise = noise
        
        # _observe determines if the observation is actually recorded from the
        #environment (true observation) or replaced by dummy data.
        #This should normally always be set to True, but can be used to temporarily
        #turn off the observation to simulate the loss of sensory signal
        self._observe = True
        
    def update(self):
        '''
        This function processes the raw image data and updates the current observation.
        '''
        # the observation is plainly the robot's camera image data
        if self._observe : observation = self.world_module.env_data['image']
        else : observation = np.zeros((self.image_dims[1], self.image_dims[0],3))
        # display the observation camera image
        if self.with_GUI:
            image_data = observation
            self.camera_image.setOpts(axisOrder='row-major')
            image_data = image_data[:,:,::-1]
            self.camera_image.setImage(image_data)
            imageScale = 1.0
            self.camera_image.setRect(QRectF(0.0, 0.0, imageScale, image_data.shape[0]/image_data.shape[1]*imageScale))
        # scale the one-line image to further reduce computational demands
        observation = cv2.resize(observation, dsize=self.image_dims)
        #resize according to field of view
        observation = crop_image(observation, self.image_dims, self.view_angle)
        observation.astype('float32')  
        observation = add_gaussian_noise(observation, self.noise[0], self.noise[1])
        observation = observation/255.0
        # display the observation camera image reduced to one line
        if self.with_GUI:
            image_data = observation
            self.observation_image.setOpts(axisOrder='row-major')
            image_data = image_data[:,:,::-1]
            self.observation_image.setImage(image_data)
            imageScale = 1.0
            self.observation_image.setRect(QRectF(0.0, -0.1, imageScale, 
                                                 image_data.shape[0]/image_data.shape[1]*imageScale))
        self.observation = observation
        
    def set_observation_state(self, state) :
        self._observe = state



class BlenderOnlineRenderer(FrontendBlenderInterface) :
    
    def teleportXY(self, objectName, x, y) : 
        xy ='%f,%f'%(x,y)
        sendStr = 'teleportXY,%s,%s'%(objectName,xy)
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)
        
    def show_object(self, objectName, state) :
        sendStr = 'show_object,%s,%s'%(objectName,state)
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)


class HexagonalGraphAllocentric(HexagonalGraph) :
    
    def generate_behavior_from_action(self, action) :
        
        callback_value = dict()
        trans_actions = np.arange(0,6)
        rot_actions   = np.arange(6,12)
        orientations = np.arange(0,360,60)
        hd_dict = dict(zip(rot_actions, orientations))
        
        if action!='reset' : 
            if action in trans_actions : 
                node_id = self.nodes[self.current_node].neighbors[action].index
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction)
            
            if action in rot_actions : 
                node_id = self.current_node
                self.head_direction = hd_dict[action]
                if node_id != -1 : 
                    self.move_to_node(node_id, self.head_direction)
                
            else  :
                self.next_node = self.current_node
                self.world_module.goalReached = True
        
        else :                 
            node_id = random.choice(self.start_nodes)
            random_direction = random.choice(orientations)
            self.head_direction = random_direction
            self.move_to_node(node_id, random_direction)
        
        self.current_node = self.next_node
        callback_value['currentNode'] = self.nodes[self.current_node]
        
        return callback_value
    
    def get_action_space(self) :
        
        return gym.spaces.Discrete(12)