/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <string.h>
#include "gstnvdsmeta.h"

#define NUM_OF_SOURCES 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 640
#define MUXER_OUTPUT_HEIGHT 360

#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

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
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  //g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added(GstChildProxy *child_proxy, GObject *object,
                      gchar *name, gpointer user_data)
{
 // g_print("Decodebin child added: %s\n", name);
  if (g_strrstr(name, "decodebin") == name)
  {
    g_signal_connect(G_OBJECT(object), "child-added",
                     G_CALLBACK(decodebin_child_added), user_data);
  }
}

static GstElement *
create_decode_bin(guint index, gchar *uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = {};

  g_snprintf(bin_name, 15, "decode-bin-%02d", index);
  /* Create a decode GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new(bin_name);

  /* decode element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin)
  {
    g_printerr("One element in decode bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the decode element */
  g_object_set(G_OBJECT(uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect(G_OBJECT(uri_decode_bin), "pad-added", G_CALLBACK(cb_newpad), bin);
  g_signal_connect(G_OBJECT(uri_decode_bin), "child-added",
                   G_CALLBACK(decodebin_child_added), bin);

  gst_bin_add(GST_BIN(bin), uri_decode_bin);

  /* We need to create a ghost pad for the decode bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad(bin, gst_ghost_pad_new_no_target("src",
                                                            GST_PAD_SRC)))
  {
    g_printerr("Failed to add ghost pad in decode bin\n");
    return NULL;
  }

  return bin;
}


int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *sink = NULL, *tiler = NULL;
  GstElement *nbxPlugin = NULL, *blender = NULL, *queue1, *queue2, *queue3, *queue4;
  GstElement *streammux[2], *nvvidconv[2];
  int i=0, fg_sources=0, bg_sources=0, j=0, num_sources=0;

  const gchar* new_mux_str = g_getenv("USE_NEW_NVSTREAMMUX");
  gboolean use_new_mux = !g_strcmp0(new_mux_str, "yes");

  GstBus *bus = NULL;
  guint bus_watch_id;

  /* Check input arguments */
  if (argc < 3) {
    g_printerr ("Usage: %s fg=<Foreground MP4 filenames> bg=<Background MP4 filenames>\n", argv[0]);
    return -1;
  }
  num_sources = argc - 1;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dsblend-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux[0] = gst_element_factory_make ("nvstreammux", "fg-streammuxer");
  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux[1] = gst_element_factory_make ("nvstreammux", "bg-streammuxer");

  /* Create nbxPlugin instance */
  nbxPlugin = gst_element_factory_make ("nvdsvideotemplate", "video-template");

  blender = gst_element_factory_make("nvblender", "blender");

  sink = gst_element_factory_make("nveglglessink", "eglsink");

  nvvidconv[0] = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter1");
  nvvidconv[1] = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
  tiler = gst_element_factory_make ("nvmultistreamtiler", "multistream-tiler");

  if (!pipeline || !streammux[0] || !streammux[1] || !nbxPlugin || !blender || !sink || !nvvidconv[0] ||!nvvidconv[1] ) {
    g_printerr ("One of elements could not be created. Exiting.\n");
    return -1;
  }

  /* Add queue elements between every two elements */
  queue1 = gst_element_factory_make("queue", "queue1");
  queue2 = gst_element_factory_make("queue", "queue2");
  queue3 = gst_element_factory_make("queue", "queue3");
  queue4 = gst_element_factory_make("queue", "queue4");

  gst_bin_add(GST_BIN(pipeline), streammux[0]);
  gst_bin_add(GST_BIN(pipeline), streammux[1]);

  for(i=0;i<num_sources;i++)
  {
    GstPad *sinkpad, *srcpad;
    GstElement *source_bin =NULL;
    gchar pad_name[16] = { };

    if(strncmp("fg=", argv[i+1], 3)==0)
    {
      g_snprintf (pad_name, 15, "sink_%u", fg_sources);
      fg_sources += 1;
      j=0;
    }
    else if(strncmp("bg=", argv[i+1], 3)==0)
    {
      g_snprintf (pad_name, 15, "sink_%u", bg_sources);
      bg_sources += 1;
      j=1;
    }
    else
    {
     g_printerr ("Usage: %s fg=<Foreground MP4 filenames> bg=<Background MP4 filenames>\n", argv[0]);
     return -1;
    }
    source_bin = create_decode_bin(i, argv[i+1]+3);

    if (!source_bin)
    {
      g_printerr("Failed to create one of source bins. Exiting.\n");
      return -1;
    }

    gst_bin_add(GST_BIN(pipeline), source_bin);

    sinkpad = gst_element_get_request_pad(streammux[j], pad_name);
    if (!sinkpad )
    {
      g_printerr("One of streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad(source_bin, "src");
    if (!srcpad)
    {
      g_printerr("Failed to get one of src pads of source bin. Exiting.\n");
      return -1;
    }

    if ( (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) )
    {
      g_printerr("Failed to link fg one of source bins to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }

  for(i=0;i<NUM_OF_SOURCES;i++)
  {
    GstPad *sinkpad = NULL, *srcpad = NULL;
    gchar pad_name[16] = {};

    g_snprintf(pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad(blender, pad_name);
    if (!sinkpad) {
      g_printerr("Failed to get nvblender background sinkpad. Exiting.\n");
      return false;
    }
    g_object_set(sinkpad, "alpha", 0.5, NULL);

    srcpad = gst_element_get_static_pad(nvvidconv[i], "src");
    if (!srcpad) {
      g_printerr("Failed to get background streammux srcpad. Exiting.\n");
      return false;
    }

    if (GST_PAD_LINK_FAILED(
            gst_pad_link(srcpad, sinkpad))) {
      g_print("AIGS Foreground video link failed. Exiting.n");
      return false;
    } else {
      g_print("AIGS Foreground Video Link succeeded.\n");
    }
    gst_object_unref(srcpad);
    gst_object_unref(sinkpad);
  }

  gst_bin_add_many(GST_BIN(pipeline), blender, sink, nbxPlugin, nvvidconv[0], nvvidconv[1], tiler, NULL);
  gst_bin_add_many(GST_BIN(pipeline), queue1, queue2, queue3, queue4, NULL);

  g_object_set (G_OBJECT (streammux[0]), "batch-size", fg_sources, NULL);
  g_object_set (G_OBJECT (streammux[1]), "batch-size", bg_sources, NULL);

  if (!use_new_mux) {
  g_object_set (G_OBJECT (streammux[0]), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  g_object_set (G_OBJECT (streammux[1]), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
  }

  g_object_set(G_OBJECT(nbxPlugin), "customlib-name", "/opt/nvidia/deepstream/deepstream/lib/libdsvfxaigs.so", NULL);
  g_object_set(G_OBJECT(nbxPlugin), "customlib-props", "enable:0 1 2", NULL);
  g_object_set(G_OBJECT(nbxPlugin), "customlib-props", "async-init:0", NULL);
  g_object_set(G_OBJECT(nbxPlugin), "customlib-props", "model-path:/usr/local/VideoFX/lib/models", NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* we link the elements together */

  if (!gst_element_link_many (streammux[0], queue1, nbxPlugin, queue2, nvvidconv[0], blender, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }
  if (!gst_element_link_many (streammux[1], queue3, nvvidconv[1], blender, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }
  if (!gst_element_link_many (blender, queue4, tiler, sink, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS (GST_BIN (pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "ds-blend");
  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s %s\n", argv[1], argv[2]);
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
