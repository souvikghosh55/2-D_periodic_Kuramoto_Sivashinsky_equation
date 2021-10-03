import numpy as np
import pickle
import matplotlib
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rc
from matplotlib.colors import LinearSegmentedColormap

try:
    import matplotlib.pyplot as plt
    
    ifplot = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    from matplotlib import ticker
    from pylab import axes
    try:
        import seaborn as sns
        
        sns.set(style='white')
        colors = [sns.xkcd_rgb['denim blue'], sns.xkcd_rgb['pale red'],
                  sns.xkcd_rgb['olive green'], sns.xkcd_rgb['golden yellow']]
    except:
        colors = ['b', 'r', 'g', 'y']
except:
    print 'Problem with matplotlib.'

matplotlib.rc('axes', labelsize=25) 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 
rc('text', usetex=True)

def initc(x, y):  # Initial condition

    u0 = np.ones((Mx, My))
    for i in range(Mx):
        for j in range(My):
            X, Y = x[i], y[j]
            u0[i,j] = np.sin(X+Y)+np.sin(X)+np.sin(Y)

    return u0

def wavenum(Mx, My):  # Wavenumber in Fourier space

    dk = np.pi/l
    kx = np.hstack((np.arange(0.,Mx/2.+1.),np.arange(-Mx/2.+1.,0.))).T*dk
    ky = np.hstack((np.arange(0.,My/2.+1.),np.arange(-My/2.+1.,0.))).T*dk

    return kx, ky 

def weights(x, y):  # Spatial integration weights 

    weights = np.zeros((Mx, My))
    nx = len(x)
    ny = len(y)
    dx = np.ones_like(x)
    dy = np.ones_like(y)

    for i in range(nx-1):
        dx[i] = x[i+1] - x[i]
        
    dx = np.delete(dx, [len(x)-1], None)
    
    for j in range(ny-1):
        dy[j] = y[j+1] - y[j]
        
    dy = np.delete(dy, [len(y)-1], None)
    
    for k in range(nx):
        for l in range(ny):
            if k == 0 and l == 0:
                weights[k,l] = dx[0]*dy[0]/4.
            elif k == 0 and l == ny-1:
                weights[k,l] = dx[0]*dy[-1]/4.
            elif k == nx-1 and l == 0:
                weights[k,l] = dx[-1]*dy[0]/4.
            elif k == nx-1 and l == ny-1:
                weights[k,l] = dx[-1]*dy[-1]/4.
            elif k == 0 and 0<l<ny-1: 
                weights[k,l] = dx[0]*(dy[l-1]+dy[l])/4.
            elif k == nx-1 and 0<l<ny-1:
                weights[k,l] = dx[-1]*(dy[l-1]+dy[l])/4.   
            elif 0<k<nx-1 and l == 0:
                weights[k,l] = (dx[k-1]+dx[k])*dy[0]/4.
            elif 0<k<nx-1 and l == ny-1:
                weights[k,l] = (dx[k-1]+dx[k])*dy[-1]/4.          
            else:
                weights[k,l] = (dx[k-1]+dx[k])*(dy[l-1]+dy[l])/4.

    return weights

def kssol(u0):  # Solves Kuramoto-Sivashinsky equation in 2-D Fourier space

    kx, ky = wavenum(Mx, My)
    nkx = len(kx)
    nky = len(ky)

    k2 = np.ones((Mx, My))
    kX = np.ones((Mx, My))
    kY = np.ones((Mx, My))
    for i in range(nkx):
        for j in range(nky):
            KX, KY = kx[i], ky[j] 
            k2[i,j] = -(KX**2)-((nu2/nu1)*(KY**2))+(nu1*((KX**4)+(2.*(nu2/nu1)*(KX**2)*(KY**2))+((nu2/nu1)**2*(KY**4))))    
            kX[i,j] = KX
            kY[i,j] = KY

    u0spec = np.fft.fft2(u0)  					# Initial condition in Fourier space

    nlinspecx = np.zeros((Mx, My, nt+1), dtype='complex')       # Nonlinear part x
    nlinspecy = np.zeros((Mx, My, nt+1), dtype='complex')       # Nonlinear part y
    nlinspecx[:,:,0] = -0.5*np.fft.fft2(np.absolute(np.fft.ifft2(1j*kX*u0spec))*np.absolute(np.fft.ifft2(1j*kX*u0spec))) 
    nlinspecy[:,:,0] = -0.5*(nu2/nu1)*np.fft.fft2(np.absolute(np.fft.ifft2(1j*kX*u0spec))*np.absolute(np.fft.ifft2(1j*kX*u0spec)))
    A = np.ones((Mx, My)) 
 
    u = np.zeros((Mx, My, nt+1), dtype='complex')		# Variable in Fourier space
    u[:,:,0] = u0spec
    ur = np.zeros((Mx, My, nt+1), dtype='complex')		# Variable in real space
    ur[:,:,0] = u0
    en = np.zeros((nt+1))					# Energy calculation
    wt = weights(x,y)
    ur2 = ur[:,:,0]*ur[:,:,0]
    en[0] = np.dot(wt.flatten(), ur2.flatten())
    nlin = np.zeros((nt+1))

    for i in range(nt):
        print i
        if i==0:
            u[:,:,i+1] = (u[:,:,i] + (dt*(nlinspecx[:,:,i] + nlinspecy[:,:,i] + (c*A*u[:,:,i]))))/(A + (dt*(k2+(c*A)))) 
            ur[:,:,i+1] = np.fft.ifft2(u[:,:,i+1]).real
            ur[:,:,i+1] = ur[:,:,i+1] - ((1./(4.*(np.pi**2)))*np.dot(wt.flatten(), ur[:,:,i+1].flatten())*A) 
            ur2 = ur[:,:,i+1]*ur[:,:,i+1]
            en[i+1] = np.dot(wt.flatten(), ur2.flatten())
        else:
            u[:,:,i] = np.fft.fft2(ur[:,:,i])
            nlinspecx[:,:,i] = -0.5*np.fft.fft2(np.absolute(np.fft.ifft2(1j*kX*u[:,:,i]))*np.absolute(np.fft.ifft2(1j*kX*u[:,:,i])))
            nlinspecy[:,:,i] = -0.5*(nu2/nu1)*np.fft.fft2(np.absolute(np.fft.ifft2(1j*kY*u[:,:,i]))*np.absolute(np.fft.ifft2(1j*kY*u[:,:,i])))
            u[:,:,i+1] = ((4.*u[:,:,i]) - u[:,:,i-1] + (4.*dt*(nlinspecx[:,:,i] + nlinspecy[:,:,i] + (c*A*u[:,:,i]))) - (2.*dt*(nlinspecx[:,:,i-1] + nlinspecy[:,:,i-1] + (c*A*u[:,:,i-1]))))/((3.*A) + (2.*dt*(k2+(c*np.ones_like(k2)))))
            ur[:,:,i+1] = np.fft.ifft2(u[:,:,i+1]).real
            ur[:,:,i+1] = ur[:,:,i+1] - ((1./(4.*(np.pi**2)))*np.dot(wt.flatten(), ur[:,:,i+1].flatten())*A)
            ur2 = ur[:,:,i+1]*ur[:,:,i+1]
            en[i+1] = np.dot(wt.flatten(), ur2.flatten())

    return ur, en
           

# Bifurcation parameters
nu1 = 0.9             							# nu1 = (pi/Lx)^2
nu2 = 0.9								# nu2 = (pi/Ly)^2

# Number of modes
Mx = 32                            					# Number of modes in x
My = 32                            					# Number of modes in y 

c = 100.                                            # Coefficient to ensure the positive definiteness of linear matrix A

# Run time 
Tf = 200.                          					# Final time
nt = 40000                         					# Number of time steps

Lx = np.pi/np.sqrt(nu1)                     				# Size of domain in x
Ly = np.pi/np.sqrt(nu2)	                     				# Size of domain in y

# Cell size
l = np.pi
dx = (2.*l)/(Mx)                    					# Grid spacing in x
dy = (2.*l)/(My) 							# Grid spacing in y

# Grid  
x = np.arange(0., Mx)*dx       						# Grid points in x
y = np.arange(0., My)*dy						# Grid points in y
X, Y = np.meshgrid(x, y, indexing='ij')					# Meshgrid in x-y
                 					
# Step-size
dt = Tf/nt                        					# Size of the time step 

t = np.linspace(0., Tf, nt+1)

u0 = initc(x,y)

ur, en = kssol(u0)

# Surface plot of the solution

colors = np.array(cm.gnuplot(np.linspace(0,1,256)))
colors[:,3] = 0.9
cmap = LinearSegmentedColormap.from_list('alpha_cmap', colors.tolist())

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, ur[:,:,-1].real, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=True, edgecolor=(0,0,0,0))
ax.contour(X, Y, ur[:,:,-1].real, 10, offset=-6, linewidth=3, cmap=cmap, linestyles="solid")
plt.gca().invert_xaxis()
plt.xticks([0., 2., 4., 6.])
plt.yticks([0., 2., 4., 6.])
ax.set_zlim(-6., 6.)
ax.set_zticks([-6., -2., 2., 6.])
ax.set_xlabel(r'$x$', fontweight='bold') 
ax.set_ylabel(r'$y$', fontweight='bold')
ax.set_zlabel(r'$v(x,y)$', rotation=90., fontweight='bold')
ax.xaxis._axinfo['label']['space_factor'] = 2.3
ax.yaxis._axinfo['label']['space_factor'] = 2.3
ax.zaxis._axinfo['label']['space_factor'] = 2.3
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.grid(linestyle='--')
plt.show()

    
