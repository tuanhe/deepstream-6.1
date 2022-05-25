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
  GstElement *pipeline = NULL, *ucxserversrc = NULL, *caps_filter = NULL,
      *nvvidconv = NULL, *nvmetaext = NULL, *nvosd = NULL, *nvvidconv2 = NULL,
      *nvenc = NULL, *h264parse = NULL, *qtmux = NULL, *filesink = NULL;
  GstCaps *caps = NULL;
  GstCapsFeatures *feature = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  gint64 port = 0;
  gchar *vidformat = "NV12";
  gint64 width = 0, height = 0;
  gchar *addr = NULL, *libpath = NULL, *output = NULL;
  gchar *endptr0 = NULL, *endptr1 = NULL, *endptr2 = NULL;

  int current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  if (argc != 7) {
    g_printerr ("Usage: %s <addr> <port> <width> <height> "
        "<video metadata serialization lib path> "
        "<file output path>\n", argv[0]);
    return -1;
  }

  addr = argv[1];
  port = g_ascii_strtoll (argv[2], &endptr0, 10);
  width = g_ascii_strtoll (argv[3], &endptr1, 10);
  height = g_ascii_strtoll (argv[4], &endptr2, 10);
  libpath = argv[5];
  output = argv[6];

  if (!g_hostname_is_ip_address (addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n",
        addr);
    return -1;
  }

  if ((port == 0 && endptr0 == argv[2]) || port <= 0) {
    g_printerr ("Incorrect port specified\n");
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

  if (libpath == NULL || output == NULL) {
    g_printerr ("Invalid custom lib path or output file specified.\n");
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-ucx-test2-server-pipeline");

  if (!pipeline) {
    g_printerr ("Pipeline could not be created. Exiting.\n");
    return -1;
  }

  /* Create the remaining elements. */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  caps_filter = gst_element_factory_make ("capsfilter", "caps_filter");
  ucxserversrc = gst_element_factory_make ("nvdsucxserversrc", "serversrc");
  nvmetaext = gst_element_factory_make ("nvdsmetaextract", "nvds-meta-extract");
  nvosd = gst_element_factory_make ("nvdsosd", "nvosd");
  nvvidconv2 =
      gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
  nvenc = gst_element_factory_make ("nvv4l2h264enc", "nvh264enc");
  h264parse = gst_element_factory_make ("h264parse", "h264-parse");
  qtmux = gst_element_factory_make ("qtmux", "qt-mux");
  filesink = gst_element_factory_make ("filesink", "file-sink");


  if (!nvvidconv || !caps_filter || !ucxserversrc || !nvmetaext ||
      !nvosd || !nvvidconv2 || !nvenc || !h264parse || !qtmux || !filesink) {
    g_printerr ("Failed to create some elements. Exiting.\n");
    return -1;
  }

  caps =
      gst_caps_new_simple ("video/x-raw",
      "format", G_TYPE_STRING, vidformat,
      "width", G_TYPE_INT, width,
      "height", G_TYPE_INT, height,
      "framerate", GST_TYPE_FRACTION, 30, 1, NULL);
  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps, 0, feature);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  g_object_set (G_OBJECT (nvmetaext), "deserialize-lib", libpath, NULL);

  g_object_set (G_OBJECT (ucxserversrc), "addr", addr, "port", port,
      "nvbuf-memory-type", 2, "num-nvbuf", 8, "buf-type", 0, NULL);

  g_object_set (G_OBJECT (filesink), "location", output, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  gst_bin_add_many (GST_BIN (pipeline), ucxserversrc, caps_filter, nvvidconv,
      nvmetaext, nvosd, nvvidconv2, nvenc, h264parse, qtmux, filesink, NULL);

  if (!gst_element_link_many (ucxserversrc, caps_filter, nvvidconv,
          nvmetaext, nvosd, nvvidconv2, nvenc, h264parse, qtmux,
          filesink, NULL)) {
    g_printerr ("Failed to link some elements. Exiting.\n");
    return -1;
  }

  g_print ("Server listening on: %s : %ld\n", addr, port);
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
