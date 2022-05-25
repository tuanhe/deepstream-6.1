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

static void
cb_newpad (GstElement * decodebin, GstPad * demux_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (demux_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *h264sink = (GstElement *) data;

  /* Need to check if the pad created by the qtdemux is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    printf ("Qtdemux src pad: %s\n", name);
    GstPad *sinkpad = gst_element_get_static_pad (h264sink, "sink");

    if (gst_pad_is_linked (sinkpad)) {
      g_printerr ("h264 parser sink pad already linked.\n");
      return;
    }

    if (gst_pad_link (demux_src_pad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link demux and h264 pads.\n");
      return;
    }

    gst_pad_set_active (demux_src_pad, TRUE);
    gst_pad_set_active (sinkpad, TRUE);

    gst_object_unref (sinkpad);
  }
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *filesrc = NULL, *streammux = NULL,
      *ucxclientsink = NULL, *qtdemux = NULL, *h264parse = NULL,
      *nvdecode = NULL, *nvvidconv = NULL, *nvinfer = NULL, *nvmetains = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  gint64 port = 0;
  gchar *addr = NULL, *inferpath = NULL, *libpath = NULL;
  gchar *endptr0 = NULL, *uri = NULL;
  GstElement *queue1 = NULL;

  int current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  if (argc != 6) {
    g_printerr ("Usage: %s <addr> <port> <infer config path> "
        "<video metadata serialization lib path> <file path>\n", argv[0]);
    return -1;
  }

  addr = argv[1];
  port = g_ascii_strtoll (argv[2], &endptr0, 10);
  inferpath = argv[3];
  libpath = argv[4];
  uri = argv[5];

  if (!g_hostname_is_ip_address (addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n",
        addr);
    return -1;
  }

  if ((port == 0 && endptr0 == argv[2]) || port <= 0) {
    g_printerr ("Incorrect port specified\n");
    return -1;
  }

  if (inferpath == NULL || libpath == NULL || uri == NULL) {
    g_printerr ("Invalid path for infer config or custom lib or "
        "file specified\n");
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-ucx-test2-server-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  gst_bin_add (GST_BIN (pipeline), streammux);

  /* Create the elements */
  filesrc = gst_element_factory_make ("filesrc", "file-src");
  qtdemux = gst_element_factory_make ("qtdemux", "qt-demux");
  h264parse = gst_element_factory_make ("h264parse", "h264-parse");
  nvdecode = gst_element_factory_make ("nvv4l2decoder", "nvv-4l2decoder");
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  nvinfer = gst_element_factory_make ("nvinfer", "nv-infer");
  nvmetains = gst_element_factory_make ("nvdsmetainsert", "nvds-meta-insert");
  ucxclientsink = gst_element_factory_make ("nvdsucxclientsink", "clientsink");
  queue1 = gst_element_factory_make ("queue", "queue1");

  if (!filesrc || !qtdemux || !h264parse || !nvdecode || !nvvidconv ||
      !nvinfer || !nvmetains || !ucxclientsink || !queue1) {
    g_printerr ("Failed to create some elements\n");
    return -1;
  }

  /* Set file src location */
  g_object_set (G_OBJECT (filesrc), "location", uri, NULL);

  /* Set the clientsink properties */
  g_object_set (G_OBJECT (ucxclientsink), "addr", addr, "port", port,
      "buf-type", 0, NULL);

  /* Set nvinfer config path */
  g_object_set (G_OBJECT (nvinfer), "config-file-path", inferpath, NULL);

  /* Set the custom lib path for metadata serialization */
  g_object_set (G_OBJECT (nvmetains), "serialize-lib", libpath, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Add elements to pipeline */
  gst_bin_add_many (GST_BIN (pipeline), filesrc, qtdemux, h264parse, nvdecode,
      nvvidconv, nvinfer, nvmetains, ucxclientsink, NULL);

  g_signal_connect (G_OBJECT (qtdemux), "pad-added",
      G_CALLBACK (cb_newpad), h264parse);

  if (!gst_element_link (filesrc, qtdemux)) {
    g_printerr ("Failed to filesrc and qtdemux. Exiting.\n");
    return -1;
  }

  if (!gst_element_link (h264parse, nvdecode)) {
    g_printerr ("Failed to h264 and nv4l2decode. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (streammux, nvvidconv, nvinfer, nvmetains,
          ucxclientsink, NULL)) {
    g_printerr ("Failed to link several elements including clientsink\n");
    return -1;
  }

  GstPad *srcpad, *muxsinkpad;
  gchar pad_name[16] = { };

  srcpad = gst_element_get_static_pad (nvdecode, "src");
  if (!srcpad) {
    g_printerr ("Failed to get src pad for decoder. Exiting\n");
    return -1;
  }

  g_snprintf (pad_name, 15, "sink_%u", 0);
  muxsinkpad = gst_element_get_request_pad (streammux, pad_name);
  if (!muxsinkpad) {
    g_printerr ("Failed to request sink pad for streammux. Exiting\n");
    return -1;
  }

  if (gst_pad_link (srcpad, muxsinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link decode src and streammux sink pads\n");
    return -1;
  }

  gst_object_unref (srcpad);
  gst_object_unref (muxsinkpad);

  g_print ("Using URI: %s\n", uri);
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
