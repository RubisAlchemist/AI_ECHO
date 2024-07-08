import bpy
import wave
import os
import numpy as np
import sys

bpy.context.scene.render.fps = 24

# Function to create a sphere mesh with shape keys for lip syncing
def create_sphere_with_shape_keys():
    # Delete default cube object
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
    sphere = bpy.context.object

    start_frame = 1
    end_frame = 130

    # Set keyframes for the location property
    sphere.location.x = 0  # Initial location
    sphere.keyframe_insert(data_path="location", index=0, frame=start_frame)  # Insert keyframe for x location at start frame
    # sphere.location.x = 5  # New location
    # sphere.keyframe_insert(data_path="location", index=0, frame=end_frame)  # Insert keyframe for x location at end frame

    # Add shape keys for different lip positions
    bpy.ops.object.shape_key_add(from_mix=False)
    bpy.context.object.active_shape_key.name = "Neutral"
    bpy.ops.object.shape_key_add(from_mix=False)
    bpy.context.object.active_shape_key.name = "Closed"
    bpy.ops.object.shape_key_add(from_mix=False)
    bpy.context.object.active_shape_key.name = "Open"

    # Adjust shape key positions as needed
    sphere.data.shape_keys.key_blocks["Closed"].value = 0.0
    sphere.data.shape_keys.key_blocks["Open"].value = 1.0

  # Switch to edit mode to select mouth vertices
    bpy.ops.object.mode_set(mode='EDIT')

    # Select vertices for the mouth
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.transform.vertex_random(offset=0.5)
    
    bpy.ops.object.mode_set(mode='OBJECT')

    # Add mouth shape key
    bpy.ops.object.shape_key_add(from_mix=False)
    bpy.context.object.active_shape_key.name = "Mouth"

    return sphere

def paint_map(object, color):
    mesh = object.data
    
    # Ensure the object has vertex colors
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    
    color_layer = mesh.vertex_colors.active.data
    
    # Check the number of vertices in the mesh
    num_vertices = len(mesh.vertices)
    
    # Iterate through each vertex and assign the color
    for vertex_idx in range(num_vertices):
        # Ensure the vertex index is within the range of the color layer
        if vertex_idx < len(color_layer):
            color_layer[vertex_idx].color = color

            

def create_anime_with_shape_keys():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    
    mesh_file = sys.argv[-1]

    bpy.ops.import_scene.obj(filepath=mesh_file)

# Get a list of all imported objects
    cam_objects = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    imported_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj!=bpy.data.objects['eye001.000'] and obj!=bpy.data.objects['eye001.001'] and obj!=bpy.data.objects['eye001.001_Sphere.001'] and obj!=bpy.data.objects['eye001.001_Sphere.005'] and obj!=bpy.data.objects['eye001.000_Sphere.006'] and obj!=bpy.data.objects['eye001.000_Sphere.001'] and obj!=bpy.data.objects['jacket']]
    print()
    imported_objects_anim = [obj for obj in bpy.data.objects if obj.type == 'MESH' and (obj==bpy.data.objects['eye001.000'] or obj==bpy.data.objects['eye001.001'] or  obj==bpy.data.objects['eye001.001_Sphere.001'] or obj==bpy.data.objects['eye001.001_Sphere.005'] or obj==bpy.data.objects['eye001.000_Sphere.006'] or obj==bpy.data.objects['eye001.000_Sphere.001'])]
    imported_objects_lips = [obj for obj in bpy.data.objects if obj.type == 'MESH' and (obj==bpy.data.objects['tongue_Cube.001'] or obj==bpy.data.objects['tongue_Cube'] or obj==bpy.data.objects['tongue'] or obj==bpy.data.objects['teeth'] or obj ==bpy.data.objects['teeth_Cube.001'] or obj== bpy.data.objects['teeth_Cube.002'])]
    imported_objects_jacket = [obj for obj in bpy.data.objects if obj.type == 'MESH' and (obj==bpy.data.objects['jacket_Cylinder.014'] or obj ==bpy.data.objects['jacket_Cylinder.001'])]

    print(cam_objects[0].location)
    print(cam_objects[0].rotation_euler)
    for obj in cam_objects:
        obj.location.x=0.1
        obj.location.y=-1.3
        obj.location.z=1.84

        obj.rotation_euler.x=1.35
        obj.rotation_euler.y=0
        obj.rotation_euler.z=0.05
        obj.keyframe_insert(data_path="location", index=-1, frame=1)  # Insert a keyframe for the location
        # obj.keyframe_insert(data_path="rotation_euler", index=-1, frame=1) 
     
    for imp in imported_objects:  #SKIN
# Set the active object to the last imported object
        bpy.context.view_layer.objects.active = imp

        # Switch to edit mode to select mouth vertices
        bpy.ops.object.mode_set(mode='EDIT')

        teeth_object = imp

        if teeth_object:
            mat = bpy.data.materials.new(name="Material")
            # Assign it to object
            if bpy.context.object.data.materials:
        # assign to 1st material slot
                bpy.context.object.data.materials[0] = mat
            else:
                bpy.context.object.data.materials.append(mat)

            mat.diffuse_color =(255.0 / 255.0, 206.0 / 255.0, 180.0 / 255.0, 1.0)

            paint_map(teeth_object, (255.0 / 255.0, 206.0 / 255.0, 180.0 / 255.0, 1.0))

    for imp in imported_objects_anim:  #EYES
# Set the active object to the last imported object
        bpy.context.view_layer.objects.active = imp

        # Switch to edit mode to select mouth vertices
        bpy.ops.object.mode_set(mode='EDIT')

        teeth_object = imp

        if teeth_object:
            mat = bpy.data.materials.new(name="Material")
            # Assign it to object
            if bpy.context.object.data.materials:
        # assign to 1st material slot
                bpy.context.object.data.materials[0] = mat
            else:
                bpy.context.object.data.materials.append(mat)

            mat.diffuse_color =(11.0 / 255.0, 156.0 / 255.0, 49.0 / 255.0, 0.6)

            paint_map(teeth_object, mat.diffuse_color)

    for imp in imported_objects_lips:  #LIPS
        bpy.context.view_layer.objects.active = imp

        # Switch to edit mode to select mouth vertices
        bpy.ops.object.mode_set(mode='EDIT')

        teeth_object = imp

        if teeth_object:
            mat = bpy.data.materials.new(name="Material")
            # Assign it to object
            if bpy.context.object.data.materials:
        # assign to 1st material slot
                bpy.context.object.data.materials[0] = mat
            else:
                bpy.context.object.data.materials.append(mat)

            mat.diffuse_color =(255.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0, 1.0)

            paint_map(teeth_object, mat.diffuse_color)


    for imp in imported_objects_jacket:  #JACKET
        bpy.context.view_layer.objects.active = imp

        # Switch to edit mode to select mouth vertices
        bpy.ops.object.mode_set(mode='EDIT')

        teeth_object = imp

        if teeth_object:
            mat = bpy.data.materials.new(name="Material")
            # Assign it to object
            if bpy.context.object.data.materials:
        # assign to 1st material slot
                bpy.context.object.data.materials[0] = mat
            else:
                bpy.context.object.data.materials.append(mat)

            mat.diffuse_color =(0.0 / 255.0, 0.0 / 255.0, 255.0 / 255.0, 1.0)

            paint_map(teeth_object, mat.diffuse_color)

    for imp in imported_objects_anim:
        bpy.context.view_layer.objects.active = imp
        bpy.ops.object.mode_set(mode='OBJECT')
        # Add shape keys for different lip positions
        bpy.ops.object.shape_key_add(from_mix=False)
        bpy.context.object.active_shape_key.name = "Neutral"
        bpy.ops.object.shape_key_add(from_mix=False)
        bpy.context.object.active_shape_key.name = "Closed"
        bpy.ops.object.shape_key_add(from_mix=False)
        bpy.context.object.active_shape_key.name = "Open"

        # Adjust shape key positions as needed
        imp.data.shape_keys.key_blocks["Closed"].value = 0
        imp.data.shape_keys.key_blocks["Open"].value = 1

        bpy.ops.object.mode_set(mode='EDIT')

        bpy.ops.mesh.select_all(action='DESELECT')

        imp.select_set(True)

        # Set the context to the object mode
        bpy.context.view_layer.objects.active = imp

        # Select all mesh elements in the active object
        #bpy.ops.mesh.select_all(action='SELECT')

        # # Apply random offset to selected vertices
        bpy.ops.transform.vertex_random(offset=2.5)

        bpy.ops.object.mode_set(mode='OBJECT')

        # Add mouth shape key
        bpy.ops.object.shape_key_add(from_mix=False)
        bpy.context.object.active_shape_key.name = "Mouth"


    return teeth_object


def resample_audio(audio_data, original_framerate, target_framerate):
    # Calculate the resampling factor
    resampling_factor = original_framerate / target_framerate
    
    # Calculate the length of resampled audio
    resampled_length = int(len(audio_data) / resampling_factor)
    
    # Resample the audio using linear interpolation
    resampled_audio = np.interp(
        np.arange(resampled_length) * resampling_factor,
        np.arange(len(audio_data)),
        audio_data
    ).astype(np.int16)
    
    return resampled_audio

# Function to animate lip syncing based on audio file
def animate_lip_sync(audio_file,scaling_factor=5000):
    # Load audio file
    # with wave.open(audio_file, 'rb') as wf:
    #     framerate = wf.getframerate()
    #     frames = wf.getnframes()
    #     duration = frames / float(framerate)

    #     amplitude_data = []
    #     while True:
    #         frame_data = wf.readframes(1)
    #         if not frame_data:
    #             break
    #         if len(frame_data) < 2:
    #             continue  # Skip frames with insufficient data
    #         amplitude = struct.unpack('<h', frame_data[:2])[0]
    #         amplitude_data.append(amplitude)


    #     print("Amplitude: ",amplitude_data[0:90])
    #     bpy.context.scene.render.fps = 24
    #     # Calculate frame duration in seconds
    #     frame_duration = 1 / bpy.context.scene.render.fps
    #     # Animate lip syncing based on amplitude data
    #     print("Duration:", duration)
    #     print("Render FPS:", bpy.context.scene.render.fps)
    #     print("Frame Duration:", frame_duration)
    #     print("Framerate:", framerate)

    #     # for frame_num in range(0, int(duration * bpy.context.scene.render.fps)):
    #     #     print("Frame Number:", frame_num)
    # # Your code inside the loop
    # print(int(duration * bpy.context.scene.render.fps))
    # min_amplitude = min(amplitude_data)
    # max_amplitude = max(amplitude_data)
    # print(min_amplitude)
    # print(max_amplitude)
    # for frame_num in range(0, int(duration * bpy.context.scene.render.fps)):
    #     if frame_num % frames >= len(amplitude_data):
    #         break
    #     bpy.context.scene.frame_set(frame_num)
    #     amplitude = (amplitude_data[frame_num % frames] - min_amplitude) / (max_amplitude - min_amplitude) - 1.0
    #     # amplitude = amplitude_data[frame_num % frames] / 32768  # Normalize amplitude
    #     scaled_amplitude = amplitude * (-800)  # Scale amplitude
    #     print("Scaled Amplitude:", scaled_amplitude)  # Print scaled amplitude
        
    with wave.open(audio_file, 'rb') as wf:
        original_framerate = wf.getframerate()
        frames = wf.getnframes()
        audio_duration = frames / float(original_framerate)
        audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)

    # Resample audio to match animation frame rate
    target_framerate = bpy.context.scene.render.fps
    resampled_audio = resample_audio(audio_data, original_framerate, target_framerate)

    # Animate lip syncing based on resampled amplitude data
    min_amplitude = np.min(resampled_audio)
    max_amplitude = np.max(resampled_audio)

    for frame_num in range(0, int(audio_duration * target_framerate)):
        if frame_num >= len(resampled_audio):
            break

        bpy.context.scene.frame_set(frame_num)
        amplitude = resampled_audio[frame_num]
        
        # Check if max_amplitude and min_amplitude are equal
        if max_amplitude == min_amplitude:
            scaled_amplitude = 0  # Set a default value
        else:
            #scaled_amplitude = ((amplitude - min_amplitude) / (max_amplitude - min_amplitude) - 1.0)
            scaled_amplitude = amplitude / 32768  # Normalize amplitude
        #print("Scaled ampl ",scaled_amplitude)
        
    # Adjust lip shape keys based on scaled amplitude
        bpy.data.objects['eye001.000'].data.shape_keys.key_blocks["Closed"].value = scaled_amplitude
        bpy.data.objects['eye001.000'].data.shape_keys.key_blocks["Open"].value = 1.0 - scaled_amplitude

        # Adjust mouth shape key
        bpy.data.objects['eye001.000'].data.shape_keys.key_blocks["Closed"].keyframe_insert(data_path="value", frame=frame_num)
        bpy.data.objects['eye001.000'].data.shape_keys.key_blocks["Open"].keyframe_insert(data_path="value", frame=frame_num)

        bpy.data.objects['eye001.000'].data.shape_keys.key_blocks["Mouth"].value = scaled_amplitude
        bpy.data.objects['eye001.000'].data.shape_keys.key_blocks["Mouth"].keyframe_insert(data_path="value", frame=frame_num)

#EYE001
        bpy.data.objects['eye001.001'].data.shape_keys.key_blocks["Closed"].value = scaled_amplitude
        bpy.data.objects['eye001.001'].data.shape_keys.key_blocks["Open"].value = 1.0 - scaled_amplitude

        # Adjust mouth shape key
        bpy.data.objects['eye001.001'].data.shape_keys.key_blocks["Closed"].keyframe_insert(data_path="value", frame=frame_num)
        bpy.data.objects['eye001.001'].data.shape_keys.key_blocks["Open"].keyframe_insert(data_path="value", frame=frame_num)

        bpy.data.objects['eye001.001'].data.shape_keys.key_blocks["Mouth"].value = scaled_amplitude
        bpy.data.objects['eye001.001'].data.shape_keys.key_blocks["Mouth"].keyframe_insert(data_path="value", frame=frame_num)

#EYE SPHERE 1
        bpy.data.objects['eye001.001_Sphere.001'].data.shape_keys.key_blocks["Closed"].value = scaled_amplitude
        bpy.data.objects['eye001.001_Sphere.001'].data.shape_keys.key_blocks["Open"].value = 1.0 - scaled_amplitude

        # Adjust mouth shape key
        bpy.data.objects['eye001.001_Sphere.001'].data.shape_keys.key_blocks["Closed"].keyframe_insert(data_path="value", frame=frame_num)
        bpy.data.objects['eye001.001_Sphere.001'].data.shape_keys.key_blocks["Open"].keyframe_insert(data_path="value", frame=frame_num)

        bpy.data.objects['eye001.001_Sphere.001'].data.shape_keys.key_blocks["Mouth"].value = scaled_amplitude
        bpy.data.objects['eye001.001_Sphere.001'].data.shape_keys.key_blocks["Mouth"].keyframe_insert(data_path="value", frame=frame_num)

#EYE SPHERE 2
        bpy.data.objects['eye001.001_Sphere.005'].data.shape_keys.key_blocks["Closed"].value = scaled_amplitude
        bpy.data.objects['eye001.001_Sphere.005'].data.shape_keys.key_blocks["Open"].value = 1.0 - scaled_amplitude

        # Adjust mouth shape key
        bpy.data.objects['eye001.001_Sphere.005'].data.shape_keys.key_blocks["Closed"].keyframe_insert(data_path="value", frame=frame_num)
        bpy.data.objects['eye001.001_Sphere.005'].data.shape_keys.key_blocks["Open"].keyframe_insert(data_path="value", frame=frame_num)

        bpy.data.objects['eye001.001_Sphere.005'].data.shape_keys.key_blocks["Mouth"].value = scaled_amplitude
        bpy.data.objects['eye001.001_Sphere.005'].data.shape_keys.key_blocks["Mouth"].keyframe_insert(data_path="value", frame=frame_num)

#EYE SPHERE 3
        bpy.data.objects['eye001.000_Sphere.006'].data.shape_keys.key_blocks["Closed"].value = scaled_amplitude
        bpy.data.objects['eye001.000_Sphere.006'].data.shape_keys.key_blocks["Open"].value = 1.0 - scaled_amplitude

        # Adjust mouth shape key
        bpy.data.objects['eye001.000_Sphere.006'].data.shape_keys.key_blocks["Closed"].keyframe_insert(data_path="value", frame=frame_num)
        bpy.data.objects['eye001.000_Sphere.006'].data.shape_keys.key_blocks["Open"].keyframe_insert(data_path="value", frame=frame_num)

        bpy.data.objects['eye001.000_Sphere.006'].data.shape_keys.key_blocks["Mouth"].value = scaled_amplitude
        bpy.data.objects['eye001.000_Sphere.006'].data.shape_keys.key_blocks["Mouth"].keyframe_insert(data_path="value", frame=frame_num)

# Function to render animation as a video file
def render_video(output_path):
    bpy.context.scene.render.resolution_x = 400
    bpy.context.scene.render.resolution_y = 200

    bpy.context.scene.cycles.samples = 30

    # Set render output format and dimensions
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'LOW'
    bpy.context.scene.render.ffmpeg.audio_codec = 'AAC'


    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 70
    # Set the output file path for the video
    bpy.context.scene.render.filepath = output_path

    # Render the video
    bpy.ops.render.render(animation=True, write_still=False)

    print("Animation rendered successfully:", output_path)

# Main function
def main():
    # Create sphere with shape keys
    sphere = create_anime_with_shape_keys()

    # Example audio file for lip syncing
    audio_file = bpy.path.abspath("//Audio.wav")

    # Animate lip syncing based on audio file
    animate_lip_sync(audio_file)

    # Render animation as a video file
    output_path = bpy.path.abspath("//output_video.mp4")
    render_video(output_path)

# Run the script
if __name__ == "__main__":
    main()
