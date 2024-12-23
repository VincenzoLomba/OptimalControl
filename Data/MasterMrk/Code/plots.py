import matplotlib.animation as animation
from matplotlib import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
import Dynamics as dyn
import numpy as np
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from matplotlib.image import BboxImage
import matplotlib.patches as patches

xDim=dyn.xDim
uDim=dyn.uDim


def plotters(xx, x_des, TT, Title): ##plot a state curve against a desired state curve

  plt.plot(xx[0,:].T, xx[1,:].T, label="Real Path", color="C1")
  if x_des is not None:
     plt.plot(x_des[0,:].T, x_des[1,:].T, label="Desired path", color="C2", linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Position")
  plt.axis('equal')
  plt.xlabel("x(m)")
  plt.ylabel("y(m)")
  plt.show()

  plt.plot(range(TT), xx[2,:].T, label="$\\psi$",color='r')
  if x_des is not None:
     plt.plot(range(TT), x_des[2,:].T, label="$\\psi$ des",color='r', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Orientation")
  plt.ylabel("$\\psi$ (rad)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()
  

  plt.plot(range(TT), xx[3,:].T, label="V",color='g')
  if x_des is not None:
     plt.plot(range(TT), x_des[3,:].T, label="V",color='g', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Velocity module")
  plt.ylabel("V (m/s)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()
  
  
  plt.plot(range(TT), xx[4,:].T, label="$\\beta$",color='y')
  if x_des is not None:
     plt.plot(range(TT), x_des[4,:].T, label="$\\beta$ des",color='y', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Velocity orientation")
  plt.ylabel("$\\beta$ (rad)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()
  
  
  plt.plot(range(TT), xx[5,:].T, label="$\\dot{\\psi}$",color='c')
  if x_des is not None:
     plt.plot(range(TT), x_des[5,:].T, label="$\\dot{\\psi}$ des",color='c', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Angular acceleration")
  plt.ylabel("$\\dot{\\psi}$ (rad/s)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()

def uplotters(uu, u_des, TT, Title): ##plot an input curve against a desired input curve

  plt.plot(range(TT-1), uu[1,:-1].T, label="$F_{x}$", color='r')
  if u_des is not None:
    plt.plot(range(TT-1), u_des[1,:-1].T, label="$F_{x}$ des", color='r',  linestyle='dashed')
    plt.legend()
  plt.grid()
  plt.title(Title + ": Thrust")
  plt.ylabel("$F_{x}$ (N)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()

  
  plt.plot(range(TT-1), uu[0,:-1].T, label="$\\delta$",color='b')
  if u_des is not None:
    plt.plot(range(TT-1), u_des[0,:-1].T, label="$\\delta$ des",color='b',  linestyle='dashed')
    plt.legend()
  plt.grid()
  plt.ylabel("$\\delta$ (rad)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.title(Title+ ": Steering angle")
  plt.show()

def globeplotters(xx1, xx2, xx3, xx4, x_des, TT, Title): ##plot state curves for different iterations against a desired state curve

  label_list=["First iteration", "Second iteration", "Fifth iteration", "Final iteration"]
  plt.plot(xx1[0,:].T, xx1[1,:].T, label=label_list[0], color='C1', alpha=0.35)
  plt.plot(xx2[0,:].T, xx2[1,:].T, label=label_list[1],color='C1', alpha=0.6)
  plt.plot(xx2[0,:].T, xx3[1,:].T, label=label_list[2],color='C1', alpha=0.8)
  plt.plot(xx3[0,:].T, xx4[1,:].T, label=label_list[3], color="C1")
  if x_des is not None:
     plt.plot(x_des[0,:].T, x_des[1,:].T, label="Desired path", linestyle="dashed", color="C2")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Position")
  plt.axis('equal')
  plt.xlabel("x(m)")
  plt.ylabel("y(m)")
  plt.show()

  plt.plot(range(TT), xx1[2,:].T, label=label_list[0],color='r', alpha=0.35)
  plt.plot(range(TT), xx2[2,:].T, label=label_list[1],color='r',alpha=0.6)
  plt.plot(range(TT), xx3[2,:].T, label=label_list[2],color='r',alpha=0.8)
  plt.plot(range(TT), xx4[2,:].T, label=label_list[3],color='r')
  if x_des is not None:
     plt.plot(range(TT), x_des[2,:].T, label="$\\psi$ des",color='r', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Orientation")
  plt.ylabel("$\\psi$ (rad)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()
  

  plt.plot(range(TT), xx1[3,:].T, label=label_list[0],color='g', alpha=0.35)
  plt.plot(range(TT), xx2[3,:].T, label=label_list[1],color='g',alpha=0.6)
  plt.plot(range(TT), xx3[3,:].T, label=label_list[2],color='g',alpha=0.8)
  plt.plot(range(TT), xx4[3,:].T, label=label_list[3],color='g')
  if x_des is not None:
     plt.plot(range(TT), x_des[3,:].T, label="$V$ des",color='g', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Velocity module")
  plt.ylabel("$V$ (m/s)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()


  plt.plot(range(TT), xx1[4,:].T, label=label_list[0],color='y', alpha=0.35)
  plt.plot(range(TT), xx2[4,:].T, label=label_list[1],color='y',alpha=0.6)
  plt.plot(range(TT), xx3[4,:].T, label=label_list[2],color='y',alpha=0.8)
  plt.plot(range(TT), xx4[4,:].T, label=label_list[3],color='y')
  if x_des is not None:
     plt.plot(range(TT), x_des[4,:].T, label="$\\beta$ des",color='y', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Velocity orientation")
  plt.ylabel("$\\beta$ (rad)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()


  plt.plot(range(TT), xx1[5,:].T, label=label_list[0],color='c', alpha=0.35)
  plt.plot(range(TT), xx2[5,:].T, label=label_list[1],color='c',alpha=0.6)
  plt.plot(range(TT), xx3[5,:].T, label=label_list[2],color='c',alpha=0.8)
  plt.plot(range(TT), xx4[5,:].T, label=label_list[3],color='c')
  if x_des is not None:
     plt.plot(range(TT), x_des[5,:].T, label="$\\dot{\\psi}$ des",color='c', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Angular velocity")
  plt.ylabel("$\\dot{\\psi}$ (rad/s)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()
  
def globeuplotters(uu1, uu2, uu3, uu4, u_des, TT, Title): ##plot input curves for different iterations against a desired input curve
   
  label_list=["First iteration", "Second iteration", "Fifth iteration", "Final iteration"]

  plt.plot(range(TT-1), uu1[1,:-1].T, label=label_list[0],color='r', alpha=0.35)
  plt.plot(range(TT-1), uu2[1,:-1].T, label=label_list[1],color='r',alpha=0.6)
  plt.plot(range(TT-1), uu3[1,:-1].T, label=label_list[2],color='r',alpha=0.8)
  plt.plot(range(TT-1), uu4[1,:-1].T, label=label_list[3],color='r')
  if u_des is not None:
     plt.plot(range(TT-1), u_des[1,:-1].T, label="$F_{x}$ des",color='r', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Thrust")
  plt.ylabel("$F_{x}$ (N)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()


  plt.plot(range(TT-1), uu1[0,:-1].T, label=label_list[0],color='b', alpha=0.35)
  plt.plot(range(TT-1), uu2[0,:-1].T, label=label_list[1],color='b',alpha=0.6)
  plt.plot(range(TT-1), uu3[0,:-1].T, label=label_list[2],color='b',alpha=0.8)
  plt.plot(range(TT-1), uu4[0,:-1].T, label=label_list[3],color='b')
  if u_des is not None:
     plt.plot(range(TT-1), u_des[0,:-1].T, label="$\\delta$ des",color='b', linestyle="dashed")
     plt.legend()
  plt.grid()
  plt.title(Title+ ": Thrust")
  plt.ylabel("$\\delta$ (rad)")
  plt.xlabel("t($10^{-2}$ s)")
  plt.show()


###############################################à
  
class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        # enlarge the image by these margins
        sx, sy = self.image_stretch 

        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent ,
                              ydescent-sy/2.5 ,
                              width,
                              height+sy)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)

        return [image]

    def set_image(self, image_path, image_stretch=(0, 0)):
        
        self.image_data = mgimg.imread(image_path) 

        self.image_stretch = image_stretch



def make_animation(xx_real,xx_opt,uu_real,TT):

  
  fig,ax=plt.subplots(figsize=(5,5))
  plt.axis([-10, 170, -80, 100])
  plt.grid()

  #setup for images inside the legend
  
  custom_handler_flame = ImageHandler()
  custom_handler_flame.set_image("./images/vertical_flame.png", image_stretch=(0, 20)) 

  custom_handler_arrow = ImageHandler()
  custom_handler_arrow.set_image("./images/arrow.png", image_stretch=(0, 20)) # this is for grace hopper
  
  ###########################
  #define line of the trajectories
  line_opt, = ax.plot(xx_opt[0,:],xx_opt[1,:], lw=2, ls='--')
  line_real, = ax.plot(xx_real[0,:],xx_real[1,:], lw=1 )

  line_flame,=ax.plot(xx_real[0,:],xx_real[1,:], color='white' )
  # reading PNG image files
  img_saetta = mgimg.imread('./images/saetta.png') 
  img_flames= mgimg.imread('./images/flames_purple.png') 

  #define velocity arrow
  velocity_arrow = patches.Arrow(xx_real[0,0], xx_real[1,0],
                            2*xx_real[3, 0]*np.cos(xx_real[2, 0]+xx_real[4,0]), 2*xx_real[3, 0]*np.sin(xx_real[2, 0]+xx_real[4,0]))
  
  # initialization function: plot the background of each frame
  def init():
    line_real.set_data(xx_real[0,0], xx_opt[1,0])
    line_opt.set_data(xx_opt[0,0], xx_opt[1,0])
    line_flame.set_data(xx_opt[0,0], xx_opt[1,0])
    ax.add_patch(velocity_arrow)
    return line_real,line_opt,velocity_arrow,
  
  def animate(frame):

    #update trajectories
    line_real.set_data(xx_real[0,:frame],xx_real[1,:frame])           
    line_opt.set_data(xx_opt[0,:frame], xx_opt[1,:frame]) 
    line_flame.set_data(xx_real[0,:frame],xx_real[1,:frame])
    
    #define transformation due to rotation of magnitude psi dot, and application of it to Saetta McQueen and flames
    tr = transforms.Affine2D().rotate_around(xx_real[0,frame],xx_real[1,frame],xx_real[2,frame])
    imobj_saetta = ax.imshow(img_saetta,transform=tr+ax.transData,extent=[xx_real[0,frame], img_saetta.shape[1]+xx_real[0,frame], xx_real[1,frame], img_saetta.shape[0]+xx_real[1,frame]], zorder=1)
    imobj_flames = ax.imshow(img_flames,transform=tr+ax.transData,extent=[xx_real[0,frame], img_flames.shape[1]+xx_real[0,frame], xx_real[1,frame], img_flames.shape[0]+xx_real[1,frame]], zorder=5)
    
    ax.imshow(img_saetta,transform=tr+ax.transData, aspect='auto')
    ax.imshow(img_flames,transform=tr+ax.transData, aspect='auto')
    
    #normalize and offsetting the force for graphic purpose, that is for rescaling the flame proportionally
    new_force=uu_real[1,:]+4000
    max_force=np.max(new_force)
    norm_force=1.5*(new_force)/max_force
    
    #velocity arrow update
    velocity_arrow=plt.Arrow(xx_real[0,frame], xx_real[1,frame], 2*xx_real[3, frame]*np.cos(xx_real[2,frame]+xx_real[4, frame]),2* xx_real[3, frame]*np.sin(xx_real[2,frame]+xx_real[4, frame]),width=7, color='limegreen')
    ax.add_patch(velocity_arrow)

    #update Saetta McQueen and flames
    imobj_saetta.set_extent([-(img_saetta.shape[1])/2 +xx_real[0,frame], (img_saetta.shape[1])/2+xx_real[0,frame], -(img_saetta.shape[0])/2 + xx_real[1,frame], (img_saetta.shape[0])/2 +xx_real[1,frame]])
    
    imobj_flames.set_extent([-(img_saetta.shape[1]) - norm_force[frame]*(img_flames.shape[1])/2 +xx_real[0,frame] +22, 22 -(img_saetta.shape[1])+xx_real[0,frame],\
                               -norm_force[frame]*(img_flames.shape[0])/4 + xx_real[1,frame], norm_force[frame]*(img_flames.shape[0])/4  +xx_real[1,frame]])
    
    
  
    return imobj_saetta,imobj_flames,line_flame,line_opt,line_real,velocity_arrow,
  
  plt.legend([line_flame,velocity_arrow, line_real,line_opt],['Module: Force Fx\nDirection: $\\psi$','Module: Velocity V\nDirection: $\\psi$+$\\beta$','Real trajectory','Optimal trajectory'],handler_map={line_flame: custom_handler_flame,velocity_arrow: custom_handler_arrow},labelspacing=1,loc='lower left',frameon=True)
  
  # call the animator, blit=True means only re-draw the parts that have changed.
  anim = animation.FuncAnimation(fig, animate,init_func=init,  frames=TT, interval=0.1, blit=True)
  
  
  plt.show()

