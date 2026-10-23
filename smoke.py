import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/genesis-reconstruction-cache')
import genesis as gs
from PIL import Image
gs.init(backend=gs.cpu,logging_level='warning')
scene=gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.Sphere(pos=(0,0,.2),radius=.2))
cam=scene.add_camera(res=(480,360),pos=(1,-1,.8),lookat=(0,0,.2),GUI=False)
scene.build()
scene.step()
rgb,*_=cam.render()
Image.fromarray(rgb).save('reconstruction/output/smoke.png')
print('SMOKE_OK',gs.__version__)
