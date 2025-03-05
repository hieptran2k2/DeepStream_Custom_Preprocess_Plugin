import sys
sys.path.append('../')
from pathlib import Path
from os import environ
import gi
import configparser
import argparse
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import time
import sys
import math
import platform
import os
from common.bus_call import bus_call
from common.FPS import PERF_DATA

import pyds
import numpy as np
import cv2

silent = False
file_loop = False
perf_data = None
measure_latency = False

MAX_DISPLAY_LEN=64
MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH=1920
TILED_OUTPUT_HEIGHT=1080
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1

# pgie_src_pad_buffer_probe  will extract metadata received on tiler sink pad
# and update params for drawing rectangle, object information etc.
def pgie_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0
    got_fps = False
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta

        if not silent:
            print("Frame Number=", frame_number, "Number of Objects=",num_rects)

        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)

        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)



def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    if file_loop:
        # use nvurisrcbin to enable file-loop
        uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def main(args, output_path, config=None):
    global perf_data
    perf_data = PERF_DATA(len(args))

    number_sources=len(args)
    
    # Standard GStreamer initialization
    Gst.init(None)

    is_live = False
    # Create pipeline form sequence element
    print("Creating Pipeline \n ")
        # ! nvdspreprocess name=preprocess \
    pipeline_desc = """
      nvstreammux name=Stream-muxer \
    ! nvdspreprocess name=preprocess \
    ! nvinfer name=primary-inference \
    ! nvdspreprocess name=preprocess-sgie \
    ! nvinfer name=second-inference \
    ! nvvideoconvert name=convertor0 \
    ! capsfilter name=capsfilter1 caps="video/x-raw(memory:NVMM), format=RGBA" \
    ! queue name= queue1 \
    ! nvmultistreamtiler name=nvtiler \
    ! nvvideoconvert name=convertor1 \
    ! queue name=queue2 \
    ! nvdsosd name=nvdsosd \
    ! nvvideoconvert name=convertor2 \
    ! capsfilter name=capsfilter2 caps="video/x-raw, format=I420" \
    ! x264enc name=encoder \
    ! h264parse \
    ! qtmux \
    ! filesink name=filesink
    """
    
    # Create pipeline
    pipeline = Gst.parse_launch(pipeline_desc)

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Streamux  
    streammux = pipeline.get_by_name("Stream-muxer")
    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    if file_loop:
        # Set nvbuf-memory-type=2 for x86 for file-loop (nvurisrcbin case)
        streammux.set_property('nvbuf-memory-type', int(pyds.NVBUF_MEM_CUDA_UNIFIED))
    if is_live:
        streammux.set_property('live-source', 1) 
        
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=args[i]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    # Preprocess
    preprocess = pipeline.get_by_name("preprocess")
    preprocess.set_property("config-file", os.path.join(config, "preprocess", "config_preprocess.txt"))

    # PGIE
    pgie = pipeline.get_by_name("primary-inference")
    pgie.set_property('config-file-path', os.path.join(config, "pgie", "config.txt"))
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        pgie.set_property("batch-size",number_sources)

    # Preprocess
    preprocess = pipeline.get_by_name("preprocess-sgie")
    preprocess.set_property("config-file", os.path.join(config, "preprocess", "config_preprocess_sgie.txt"))
    
    # SGIE
    sgie = pipeline.get_by_name("second-inference")
    sgie.set_property('config-file-path', os.path.join(config, "sgie", "config.txt")) 

    # Convertor
    convertor = pipeline.get_by_name("convertor0")
    convertor.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))
    
    # Tiler
    tiler = pipeline.get_by_name("nvtiler")
    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    tiler.set_property("compute-hw", 1)
    tiler.set_property("nvbuf-memory-type", int(pyds.NVBUF_MEM_CUDA_UNIFIED))
    tiler_src_pad = tiler.get_static_pad("sink")
    tiler_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_sink_pad_buffer_probe, 0)
    # Perf callback function to print fps every 5 sec
    GLib.timeout_add(5000, perf_data.perf_print_callback)
        
    # Encoder
    encoder = pipeline.get_by_name("encoder")
    encoder.set_property("bitrate", 2000000)
    
    # Nvdsosd
    nvosd = pipeline.get_by_name("nvdsosd")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    
    # Filesink
    sink = pipeline.get_by_name("filesink")
    sink.set_property("location", output_path)
    sink.set_property("sync", 1)
    sink.set_property("async", 0)
    sink.set_property("qos",0)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Enable latency measurement via probe if environment variable NVDS_ENABLE_LATENCY_MEASUREMENT=1 is set.
    # To enable component level latency measurement, please set environment variable
    # NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1 in addition to the above.
    if environ.get('NVDS_ENABLE_LATENCY_MEASUREMENT') == '1':
        print ("Pipeline Latency Measurement enabled!\nPlease set env var NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1 for Component Latency Measurement")
        global measure_latency
        measure_latency = True

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events		
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def parse_args():

    parser = argparse.ArgumentParser(prog="deepstream_test",
                    description="deepstream-test3 multi stream, multi model inference reference app")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input streams",
        nargs="+",
        metavar="URIs",
        default=["a"],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output file",
        default="path/to/output/out.mp4",
    )
    parser.add_argument(
        "-c",
        "--configfile",
        default="/path/to/folder/model/",
        help="Choose the config-file to be used with specified pgie",
    )
    parser.add_argument(
        "--file-loop",
        action="store_true",
        default=False,
        dest='file_loop',
        help="Loop the input file sources after EOS",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        default=False,
        dest='silent',
        help="Disable verbose output",
    )
    # Check input arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    stream_paths = args.input
    output_path = args.output
    config = args.configfile
    global silent
    global file_loop
    silent = args.silent
    file_loop = args.file_loop

    if config:
        config_path = Path(config)
        if not config_path.is_dir():
            sys.stderr.write ("Specified config-file: %s doesn't exist. Exiting...\n\n" % config)
            sys.exit(1)

    print(vars(args))
    return stream_paths, output_path, config

if __name__ == '__main__':
    stream_paths, output_path, config  = parse_args()
    sys.exit(main(stream_paths, output_path, config ))

