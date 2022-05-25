/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "nvdsinfer_custom_impl.h"
#include "gstnvinferaudio_custom_parser.h"
#include <cstring>

constexpr bool enable_coarse_label_saving = false;

std::vector<unsigned int> index_giver_subcategory(const char *labell)
{
    auto label_str = std::string(labell);
    if (label_str == "1_engine")
        return std::vector<unsigned int>{8, 9, 10};
    if (label_str == "2_machinery-impact")
        return std::vector<unsigned int>{11, 12, 13, 14};
    if (label_str == "3_non-machinery-impact")
        return std::vector<unsigned int>{15};
    if (label_str == "4_powered-saw")
        return std::vector<unsigned int>{16, 17, 18};
    if (label_str == "5_alert-signal")
        return std::vector<unsigned int>{19, 20, 21, 22};
    if (label_str == "6_music")
        return std::vector<unsigned int>{23, 24, 25};
    if (label_str == "7_human-voice")
        return std::vector<unsigned int>{26, 27, 28, 29};
    if (label_str == "8_dog")
        return std::vector<unsigned int>{30};
    return std::vector<unsigned int>{};
}

extern "C"
{
    bool NvDsInferParseCustomAudio(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo const &networkInfo, float classifierThreshold,
                                   std::vector<NvDsInferAttribute> &attrList, std::string &attrString)
    {

        float *outputCoverageBuffer = static_cast<float *>(outputLayersInfo[0].buffer);
        float max_probability_coarse = 0;
        bool coarse_attr_found = false;
        NvDsInferAttribute coarse_attr;

        /* Iterate through all the probabilities that the object belongs to
         * each COARSE class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < NB_COARSE_LABEL_AUDIO; c++)
        {
            float probability = outputCoverageBuffer[c];
            if (probability > classifierThreshold && probability > max_probability_coarse)
            {
                max_probability_coarse = probability;
                coarse_attr_found = true;
                coarse_attr.attributeIndex = 0;
                coarse_attr.attributeValue = c;
                coarse_attr.attributeConfidence = probability;
            }
        }
        if (coarse_attr_found)
        {
            coarse_attr.attributeLabel = strdup(LABELS_AUDIO.begin()[coarse_attr.attributeValue]);
            /// Change enable_coarse_label_saving to (en/dis)able the saving of coarse element
            if (enable_coarse_label_saving)
                attrList.push_back(coarse_attr);
            const auto vect = index_giver_subcategory(coarse_attr.attributeLabel);
            NvDsInferAttribute fine_attr;
            float max_probability_fine = 0.0;
            for (const auto &index : vect)
            {
                float probability = outputCoverageBuffer[index];
                if (probability > max_probability_fine)
                {
                    max_probability_fine = probability;
                    fine_attr.attributeIndex = 0;
                    fine_attr.attributeValue = index;
                    fine_attr.attributeConfidence = probability;
                }
            }
            fine_attr.attributeLabel = strdup(LABELS_AUDIO.begin()[fine_attr.attributeValue]);
            attrList.push_back(fine_attr);
            if (enable_coarse_label_saving){
                attrString.append("(Coarse: ").append(coarse_attr.attributeLabel);
                attrString.append(", Fine: ").append(fine_attr.attributeLabel).append(")");
            } else
                attrString.append("(Fine: ").append(fine_attr.attributeLabel).append(")");
        }
        return true;
    }
}
