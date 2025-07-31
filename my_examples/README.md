First you need to create the templates of the CAD Block with the icosahedron.py file. You just need to include the CAD Path in:
```
mesh = trimesh.load("/home/skoumal/dev/BlenderProc/LegoBlock_full.ply")
```
Then you get the RGB, depth and camera_pose files in a folder called:
```
output_dir = "lego_views"
```
Then go to the create_pcd_cad.py file and change the path:
```
folder = "/home/skoumal/dev/TEASER-plusplus/build/python/lego_views"
```
To the path where you stored the images and depth data. Then run it.
Finally you can run the detection_final.py file for robust teaser++ registration. It should work. Just change these paths
```
    depth_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000005.png"
    rgb_path   = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000005.jpg"
    mask_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/mask/000005_000000.png"
    scene_camera_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data/Legoblock/train_pbr/000000/scene_camera.json"
    
    #Read CAD Path
    cad_path = "/home/skoumal/dev/TEASER-plusplus/build/python/lego_views/"
  ```
