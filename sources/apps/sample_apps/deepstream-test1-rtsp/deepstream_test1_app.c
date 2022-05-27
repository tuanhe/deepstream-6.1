/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include <gst/rtsp-server/rtsp-server.h>

#define RTSP 1

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};

/* osd_sink_pad_buffer_probe  will extract metadata received on OSD sink pad
 * and update params for drawing rectangle, object information etc. */

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }

    g_print ("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}


static GstRTSPServer *server;
static gboolean
start_rtsp_streaming (guint rtsp_port_num, guint updsink_port_num,
                      guint64 udp_buffer_size)
{
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;
    char udpsrc_pipeline[512];

    char port_num_Str[64] = { 0 };
    char *encoder_name;

    if (udp_buffer_size == 0)
        udp_buffer_size = 512 * 1024;

    sprintf (udpsrc_pipeline,
             "( udpsrc name=pay0 port=%d buffer-size=%lu caps=\"application/x-rtp, media=video, "
             "clock-rate=90000, encoding-name=H264, payload=96 \" )",
             updsink_port_num, udp_buffer_size);

    sprintf (port_num_Str, "%d", rtsp_port_num);

    server = gst_rtsp_server_new ();
    g_object_set (server, "service", port_num_Str, NULL);

    mounts = gst_rtsp_server_get_mount_points (server);

    factory = gst_rtsp_media_factory_new ();
    gst_rtsp_media_factory_set_launch (factory, udpsrc_pipeline);

    gst_rtsp_mount_points_add_factory (mounts, "/ds-test", factory);

    g_object_unref (mounts);

    gst_rtsp_server_attach (server, NULL);

    g_print
            ("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n",
             rtsp_port_num);

    return TRUE;
}

static void rtsp_server_init(GstElement* rtsp_element, GstElement* sink)
{
  guint udp_port  = 5400;
  guint rtsp_port = 8554;
  g_object_set (G_OBJECT (rtsp_element), "preset-level", 1, NULL);
  g_object_set (G_OBJECT (rtsp_element), "insert-sps-pps", 1, NULL);
  g_object_set (G_OBJECT (rtsp_element), "bufapi-version", 1, NULL);
  g_object_set (G_OBJECT (sink), "host", "127.0.0.1", "port",
                udp_port, "async", FALSE, "sync", 1, NULL);
  start_rtsp_streaming (rtsp_port, udp_port, 0);
}

static GstRTSPFilterResult
client_filter (GstRTSPServer * server, GstRTSPClient * client,
               gpointer user_data)
{
    return GST_RTSP_FILTER_REMOVE;
}

static void destroy_sink_bin ()
{
  GstRTSPMountPoints *mounts;
  GstRTSPSessionPool *pool;

  mounts = gst_rtsp_server_get_mount_points (server);
  gst_rtsp_mount_points_remove_factory (mounts, "/ds-test");
  g_object_unref (mounts);
  gst_rtsp_server_client_filter (server, client_filter, NULL);
  pool = gst_rtsp_server_get_session_pool (server);
  gst_rtsp_session_pool_cleanup (pool);
  g_object_unref (pool);
}


int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL;
  
  GstElement *nvvidconv_postosd = NULL, *filter = NULL,
             *encoder = NULL, *rtppay = NULL;

  GstElement *transform = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  GstPad *osd_sink_pad = NULL;

  int current_device = -1;
  cudaGetDevice(&current_device);
  g_print("current device : %d\n", current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);
  /* Check input arguments */
  if (argc != 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    g_printerr ("OR: %s <H264 filename>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest1-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  #if RTSP
  nvvidconv_postosd = gst_element_factory_make ("nvvideoconvert", "nvvidconv_postosd");

  filter = gst_element_factory_make ("capsfilter", "filter");
      GstCaps* caps4 = gst_caps_from_string ("video/x-raw, format=I420");
      g_object_set (G_OBJECT (filter), "caps", caps4, NULL);
      gst_caps_unref (caps4);

  encoder = gst_element_factory_make ("encoder", "h264 encoder");

  rtppay = gst_element_factory_make ("rtph264pay", "rtp-payer");
  
  sink = gst_element_factory_make ("udpsink", "udp-sink");
  
  if (!nvvidconv_postosd || !filter || !encoder || !rtppay) {
      g_printerr ("One element could not be created.%p,%p,%p,%p Exiting.\n",
                                                    nvvidconv_postosd,
                                                    filter,
                                                    encoder, 
                                                    rtppay);
      return -1;}
  /* Finally render the osd output */
  if(prop.integrated) {
      transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
  }


  #else
  /* Finally render the osd output */
  if(prop.integrated) {
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
  }
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  #endif

  if (!source || !h264parser || !decoder || !pgie
      || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  if(!transform && prop.integrated) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }

  /* we set the input filename to the source element */
  g_object_set (G_OBJECT (source), "location", argv[1], NULL);

  if (g_str_has_suffix (argv[1], ".h264")) {
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);

    g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
        MUXER_OUTPUT_HEIGHT,
        "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set (G_OBJECT (pgie),
        "config-file-path", "dstest1_pgie_config.txt", NULL);
  }

  if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {

    nvds_parse_file_source(source, argv[1],"source");
    nvds_parse_streammux(streammux, argv[1],"streammux");

    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
    g_object_set (G_OBJECT (pgie),
        "config-file-path", "dstest1_pgie_config.yml", NULL);
  }

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  if(prop.integrated) {
    gst_bin_add_many (GST_BIN (pipeline),
        source, h264parser, decoder, streammux, pgie,
        nvvidconv, nvosd, transform, sink, NULL);
  }
  else {
  #if RTSP
  gst_bin_add_many (GST_BIN (pipeline),
        source, h264parser, decoder, streammux, pgie, 
        nvvidconv, nvosd, 
        nvvidconv_postosd, filter, encoder, rtppay, sink, NULL);
  #else
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux, pgie,
      nvvidconv, nvosd, sink, NULL);
  #endif
  }

  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (decoder, pad_name_src);
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */

  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  if(prop.integrated) {
    if (!gst_element_link_many (streammux, pgie,
        nvvidconv, nvosd, transform, sink, NULL)) {
      g_printerr ("Elements could not be linked: 2. Exiting.\n");
      return -1;
    }
  }
  else {
  #if RTSP
    if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, 
                                nvvidconv_postosd, filter, encoder, rtppay, sink,NULL)) {
            g_printerr ("Elements could not be linked: 3. Exiting.\n");
            return -1;
    }
  #else  
    if (!gst_element_link_many (streammux, pgie,
        nvvidconv, nvosd, sink, NULL)) {
      g_printerr ("Elements could not be linked: 2. Exiting.\n");
      return -1;
    }
  #endif 
  }

  /* rtsp init */
  rtsp_server_init(encoder, sink);

  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Set the pipeline to "playing" state */
  g_print ("Using file: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  destroy_sink_bin();
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
