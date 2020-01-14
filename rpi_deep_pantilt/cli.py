# -*- coding: utf-8 -*-

"""Console script for rpi_deep_pantilt."""
import logging
import sys
import time
import click

import numpy as np

import cv2

# from rpi_deep_pantilt.detect.camera import PiCameraStream
from rpi_deep_pantilt.detect.ssd_mobilenet_v3_coco import SSDMobileNet_V3_Small_Coco_PostProcessed, SSDMobileNet_V3_Coco_EdgeTPU_Quant
from rpi_deep_pantilt.control.manager import pantilt_process_manager
from rpi_deep_pantilt.control.hardware_test import pantilt_test, camera_test


@click.group()
def cli():
    pass


def run_detect(vs, model):
    LOGLEVEL = logging.getLogger().getEffectiveLevel()

    start_time = time.time()
    fps_counter = 0
    stream_name = "Stream object detection"
    while True:
        read, frame = vs.read()
        
        if frame is None:
            break

        #model requires 320x320 and has [352, 288]
        required_res = (320, 320)
               
        #FOR UPSCALE
#         scale_percent = required_res[0] * 100/ frame.shape[0]
#         
#         height = required_res[0]
#         width = int(frame.shape[1] * scale_percent / 100)
#         
#         # resize image
#         resized_frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
#        
#         #crop frame to correct dimensions
#         cropped_frame = resized_frame[:, 0:required_res[1]]


        #DOWNSCALE AND PATCHING
        scale_percent = required_res[0] * 100/ frame.shape[1]
        
        width = required_res[0]
        height = int(frame.shape[0] * scale_percent / 100)
        
        # resize image
        resized_frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
       
        #crop frame to correct dimensions
        bordersize = required_res[1] - height

        cropped_frame = cv2.copyMakeBorder(
            resized_frame,
            top=0,
            left=0,
            right=0,
            bottom=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        
        prediction = model.predict(cropped_frame)
        overlay = model.create_overlay(
            frame, prediction)
#         capture_manager.overlay_buff = overlay
        cv2.imshow(stream_name, overlay)
        
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

        
        if LOGLEVEL <= logging.INFO:
            fps_counter += 1
            if (time.time() - start_time) > 1:
                fps = fps_counter / (time.time() - start_time)
                logging.info(f'FPS: {fps}')
                fps_counter = 0
                start_time = time.time()


@cli.command()
@click.option('--loglevel', required=False, type=str, default='WARNING', help='Run object detection without pan-tilt controls. Pass --loglevel=DEBUG to inspect FPS.')
@click.option('--edge-tpu', is_flag=True, required=False, type=bool, default=False, help='Accelerate inferences using Coral USB Edge TPU')
def detect(loglevel, edge_tpu):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)

    if edge_tpu:
        model = SSDMobileNet_V3_Coco_EdgeTPU_Quant()
    else:
        model = SSDMobileNet_V3_Small_Coco_PostProcessed()

#     capture_manager = PiCameraStream(resolution=(320, 320))
#     capture_manager.start()
#     capture_manager.start_overlay()

    resolution=[352, 288]
    framerate=20
    # Open camera
    vs = cv2.VideoCapture(0)
    if not vs.isOpened():
        print("Cannot open camera")
        exit()

    # Init resolution
    resolution = tuple(resolution)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    try:
        run_detect(vs, model)
    except KeyboardInterrupt:
        vs.release()


@cli.command()
@click.option('--loglevel', required=False, type=str, default='WARNING', help='List all valid classification labels')
def list_labels(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    model = SSDMobileNet_V3_Small_Coco_PostProcessed()
    print('You can detect / track the following objects:')
    print([x['name'] for x in model.category_index.values()])


@cli.command()
@click.option('--label', required=True, type=str, default='person', help='The class label to track, e.g `orange`. Run `rpi-deep-pantilt list-labels` to inspect all valid values')
@click.option('--loglevel', required=False, type=str, default='WARNING')
@click.option('--edge-tpu', is_flag=True, required=False, type=bool, default=False, help='Accelerate inferences using Coral USB Edge TPU')
def track(label, loglevel, edge_tpu):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return pantilt_process_manager(edge_tpu=edge_tpu, labels=(label,))


@cli.group()
def test():
    pass


@test.command()
@click.option('--loglevel', required=False, type=str, default='INFO')
def pantilt(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return pantilt_test()


@test.command()
@click.option('--loglevel', required=False, type=str, default='INFO')
def camera(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    return camera_test()


def main():
    cli()


if __name__ == "__main__":
    main()
