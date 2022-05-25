/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

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

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *ucxclientsrc = NULL, *nvvidconv = NULL,
    *caps_filter = NULL, *filesink = NULL, *queue1 = NULL;
  GstElement *h264enc = NULL, *h264parser = NULL, *qtmux = NULL;

  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  gint64 port = 0;
  gchar *addr = NULL;
  gchar *vidformat = "NV12";
  gint64 width = 0, height = 0;
  gchar *endptr0 = NULL, *endptr1 = NULL, *endptr2 = NULL, *outfile = NULL;

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc != 6) {
    g_printerr ("Usage: %s <addr> <port> <width> <height> <outputfile>\n", argv[0]);
    return -1;
  }

  addr = argv[1];
  port = g_ascii_strtoll (argv[2], &endptr0, 10);
  width = g_ascii_strtoll (argv[3], &endptr1, 10);
  height = g_ascii_strtoll (argv[4], &endptr2, 10);
  outfile = argv[5];

  if (!g_hostname_is_ip_address (addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n", addr);
    return -1;
  }

  if ((port == 0 && endptr0 == argv[2]) ||
      (width == 0 && endptr1 == argv[3]) ||
      (height == 0 && endptr2 == argv[4])) {
    g_printerr ("Incorrect port, width or height specified\n");
    return -1;
  }

  if (port <= 0 || width <= 0 || height <= 0) {
    g_printerr ("Invalid port, width or height\n");
    return -1;
  }

  if (outfile == NULL) {
    g_printerr ("Invalid output file\n");
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-ucx-test1-client-pipeline");

  ucxclientsrc = gst_element_factory_make ("nvdsucxclientsrc", "ucxclientsrc");
  caps_filter = gst_element_factory_make ("capsfilter", "caps_filter");
  queue1 = gst_element_factory_make ("queue", "queue1");
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  h264enc = gst_element_factory_make ("nvv4l2h264enc", "h264-hw-enc");
  h264parser = gst_element_factory_make ("h264parse", "h264-parse");
  qtmux = gst_element_factory_make ("qtmux", "qt-mux");
  filesink = gst_element_factory_make ("filesink", "file-sink");

  if (!ucxclientsrc || !caps_filter || !queue1 || !nvvidconv || !h264enc ||
      !h264parser || !qtmux || !filesink) {
    g_printerr ("One pipeline element could not be created. Exiting\n");
    return -1;
  }

  /* Set ucxclientsrc properties */
  g_object_set (G_OBJECT (ucxclientsrc), "addr", addr, "port", port,
      "nvbuf-batch-size", 1, "num-nvbuf", 4, "nvbuf-memory-type", 2, NULL);

  caps =
      gst_caps_new_simple ("video/x-raw",
      "format", G_TYPE_STRING, vidformat,
      "width", G_TYPE_INT, width,
      "height", G_TYPE_INT, height,
      "framerate", GST_TYPE_FRACTION, 30, 1,
      NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  /* Set filesink properties */
  g_object_set (G_OBJECT (filesink), "location", outfile, "async", 0,
      "sync", 1, "qos", 0, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  gst_bin_add_many (GST_BIN (pipeline), ucxclientsrc, caps_filter,
      nvvidconv, queue1, h264enc, h264parser, qtmux, filesink, NULL);

  if (!gst_element_link_many (ucxclientsrc, caps_filter, queue1, nvvidconv,
          h264enc, h264parser, qtmux, filesink, NULL)) {
    g_printerr ("Failed to link several elements. Exiting.\n");
    return -1;
  }

  /* Set the pipeline to "playing" state */
  g_print ("Now saving stream to: %s\n", argv[5]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  return 0;
}
