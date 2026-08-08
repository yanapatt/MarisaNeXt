# [MarisaNeXt: Mushroom Accurate Recognition via Integrated Simple Attention based on ConvNeXt](#)

## Performance results of the ablation study on the test set with Soft-voting ensemble
<table>
  <thead>
    <tr>
      <th rowspan="2">Attention Layer</th>
      <th rowspan="2">Model</th>
      <th rowspan="2">Batch Size</th>
      <th rowspan="2">Image Size</th>
      <th rowspan="2">GFLOPs</th>
      <th rowspan="2">GMACs</th>
      <th rowspan="2">#Params (M)</th>
      <th colspan="9">Mushroom A</th>
      <th colspan="9">Mushroom B</th>
    </tr>
    <tr>
      <th>Latency (ms/img)</th><th>Latency (ms/batch)</th><th>Throughput (img/s)</th><th>Loss</th><th>Acc</th><th>F1</th><th>Precision</th><th>Recall</th><th>Link to File</th>
      <th>Latency (ms/img)</th><th>Latency (ms/batch)</th><th>Throughput (img/s)</th><th>Loss</th><th>Acc</th><th>F1</th><th>Precision</th><th>Recall</th><th>Link to File</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>None</td><td>ConvNeXt-Tiny</td><td>32</td><td>224×224</td><td>8.9379</td><td>4.4548</td><td>27.8</td>
      <td>1.7638</td><td>55.6858</td><td>566.9816</td><td>0.1009</td><td>0.9698</td><td>0.9678</td><td>0.9712</td><td>0.9645</td><td><a href="./notebooks/MushroomA/ConvNeXt_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.8026</td><td>56.9235</td><td>554.8174</td><td>0.0516</td><td>0.9900</td><td>0.9890</td><td>0.9900</td><td>0.9884</td><td><a href="./notebooks/MushroomB/ConvNeXt_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CBAM</td><td>Model-A1</td><td>32</td><td>224×224</td><td>8.9393</td><td>4.4555</td><td>28.0</td>
      <td>1.7924</td><td>56.5902</td><td>557.8978</td><td>0.1159</td><td>0.9691</td><td>0.9670</td><td>0.9690</td><td>0.9651</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_A1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.7721</td><td>55.9606</td><td>564.3154</td><td>0.0537</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_A1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CBAM</td><td>Model-A2</td><td>32</td><td>224×224</td><td>8.9450</td><td>4.4583</td><td>29.4</td>
      <td>2.0342</td><td>64.2210</td><td>491.6119</td><td>0.1104</td><td>0.9713</td><td>0.9689</td><td>0.9691</td><td>0.9687</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_A2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.9798</td><td>62.5197</td><td>505.1087</td><td>0.0517</td><td>0.9883</td><td>0.9863</td><td>0.9869</td><td>0.9862</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_A2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CBAM</td><td>Model-A3</td><td>32</td><td>224×224</td><td>8.9393</td><td>4.4555</td><td>28.0</td>
      <td>1.8861</td><td>59.5466</td><td>530.2043</td><td>0.1072</td><td>0.9698</td><td>0.9687</td><td>0.9732</td><td>0.9644</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_A3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.7683</td><td>55.8406</td><td>565.5254</td><td>0.0566</td><td>0.9900</td><td>0.9892</td><td>0.9898</td><td>0.9892</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_A3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SE</td><td>Model-B1</td><td>32</td><td>224×224</td><td>8.9387</td><td>4.4549</td><td>28.0</td>
      <td>1.8120</td><td>57.2070</td><td>551.9190</td><td>0.1158</td><td>0.9683</td><td>0.9683</td><td>0.9718</td><td>0.9650</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_B1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.7619</td><td>55.6386</td><td>567.5798</td><td>0.0560</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_B1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SE</td><td>Model-B2</td><td>32</td><td>224×224</td><td>8.9433</td><td>4.4563</td><td>29.4</td>
      <td>1.7754</td><td>56.0519</td><td>563.2819</td><td>0.1067</td><td>0.9698</td><td>0.9672</td><td>0.9679</td><td>0.9667</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_B2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.8640</td><td>58.8630</td><td>536.4939</td><td>0.0512</td><td>0.9900</td><td>0.9893</td><td>0.9897</td><td>0.9893</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_B2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SE</td><td>Model-B3</td><td>32</td><td>224×224</td><td>8.9387</td><td>4.4549</td><td>28.0</td>
      <td>1.6942</td><td>53.4892</td><td>590.3079</td><td>0.0990</td><td>0.9698</td><td>0.9678</td><td>0.9722</td><td>0.9638</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_B3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.7225</td><td>54.3950</td><td>580.5541</td><td>0.0540</td><td>0.9900</td><td>0.9889</td><td>0.9896</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_B3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CA</td><td>Model-C1</td><td>32</td><td>224×224</td><td>8.9462</td><td>4.4589</td><td>28.0</td>
      <td>1.8514</td><td>58.4515</td><td>540.1599</td><td>0.1295</td><td>0.9638</td><td>0.9617</td><td>0.9650</td><td>0.9586</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_C1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.8192</td><td>57.4479</td><td>549.7118</td><td>0.0601</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_C1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CA</td><td>Model-C2</td><td>32</td><td>224×224</td><td>9.0000</td><td>4.4857</td><td>30.2</td>
      <td>2.2211</td><td>70.1218</td><td>450.2402</td><td>0.1151</td><td>0.9698</td><td>0.9694</td><td>0.9709</td><td>0.9679</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_C2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.2423</td><td>70.8091</td><td>445.9778</td><td>0.0522</td><td>0.9883</td><td>0.9868</td><td>0.9879</td><td>0.9862</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_C2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CA</td><td>Model-C3</td><td>32</td><td>224×224</td><td>8.9462</td><td>4.4589</td><td>28.0</td>
      <td>1.8551</td><td>58.5682</td><td>539.0953</td><td>0.1104</td><td>0.9683</td><td>0.9660</td><td>0.9698</td><td>0.9625</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_C3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.8385</td><td>58.0577</td><td>543.9278</td><td>0.0568</td><td>0.9883</td><td>0.9874</td><td>0.9884</td><td>0.9869</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_C3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D1</td><td>32</td><td>224×224</td><td>8.9379</td><td>4.4548</td><td>27.8</td>
      <td>1.8376</td><td>58.0166</td><td>544.1945</td><td>0.1038</td><td>0.9713</td><td>0.9684</td><td>0.9703</td><td>0.9666</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.8251</td><td>57.6341</td><td>547.9230</td><td>0.0511</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D2</td><td>32</td><td>224×224</td><td>8.9379</td><td>4.4548</td><td>27.8</td>
      <td>2.2484</td><td>70.9862</td><td>444.7628</td><td>0.0976</td><td>0.9751</td><td>0.9730</td><td>0.9757</td><td>0.9704</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.2184</td><td>70.0546</td><td>450.7798</td><td>0.0558</td><td>0.9900</td><td>0.9889</td><td>0.9896</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D3</td><td>32</td><td>224×224</td><td>8.9379</td><td>4.4548</td><td>27.8</td>
      <td>1.8142</td><td>57.2783</td><td>551.2013</td><td>0.1065</td><td>0.9706</td><td>0.9689</td><td>0.9709</td><td>0.9669</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.8346</td><td>57.9354</td><td>545.0745</td><td>0.0528</td><td>0.9917</td><td>0.9911</td><td>0.9915</td><td>0.9910</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td><b>SimAM</b></td><td><b>Model-D4</b></td><td><b>32</b></td><td><b>224×224</b></td><td><b>8.9379</b></td><td><b>4.4548</b></td><td><b>27.8</b></td>
      <td><b>2.3132</b></td><td><b>73.0312</b></td><td><b>432.3013</b></td><td><b>0.0997</b></td><td><b>0.9729</b></td><td><b>0.9694</b></td><td><b>0.9714</b></td><td><b>0.9675</b></td><td><b><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomA_R1_V1.ipynb">Link</a></b></td>
      <td><b>2.3330</b></td><td><b>73.6735</b></td><td><b>428.6468</b></td><td><b>0.0528</b></td><td><b>0.9917</b></td><td><b>0.9912</b></td><td><b>0.9917</b></td><td><b>0.9910</b></td><td><b><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomB_R1_V1.ipynb">Link</a></b></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D5</td><td>32</td><td>224×224</td><td>8.9379</td><td>4.4548</td><td>27.8</td>
      <td>2.3053</td><td>72.7817</td><td>433.7837</td><td>0.1052</td><td>0.9744</td><td>0.9720</td><td>0.9759</td><td>0.9683</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D5_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.3181</td><td>73.2039</td><td>431.3880</td><td>0.0548</td><td>0.9900</td><td>0.9892</td><td>0.9898</td><td>0.9892</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D5_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D6</td><td>32</td><td>224×224</td><td>8.9379</td><td>4.4548</td><td>27.8</td>
      <td>2.3869</td><td>75.3587</td><td>418.9502</td><td>0.1083</td><td>0.9691</td><td>0.9649</td><td>0.9674</td><td>0.9625</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D6_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.4096</td><td>76.0916</td><td>415.0135</td><td>0.0517</td><td>0.9900</td><td>0.9892</td><td>0.9898</td><td>0.9892</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D6_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
  </tbody>
</table>

## Performance results of various models and MarisaNeXt on the test set with Soft-voting ensemble
<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Batch Size</th>
      <th rowspan="2">Image Size</th>
      <th rowspan="2">GFLOPs</th>
      <th rowspan="2">GMACs</th>
      <th rowspan="2">#Params (M)</th>
      <th colspan="9">Mushroom A</th>
      <th colspan="9">Mushroom B</th>
    </tr>
    <tr>
      <th>Latency (ms/img)</th><th>Latency (ms/batch)</th><th>Throughput (img/s)</th><th>Loss</th><th>Acc</th><th>F1</th><th>Precision</th><th>Recall</th><th>Link to File</th>
      <th>Latency (ms/img)</th><th>Latency (ms/batch)</th><th>Throughput (img/s)</th><th>Loss</th><th>Acc</th><th>F1</th><th>Precision</th><th>Recall</th><th>Link to File</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet50</td><td>32</td><td>224×224</td><td>8.2070</td><td>4.0872</td><td>23.5</td>
      <td>1.4484</td><td>45.7271</td><td>690.4333</td><td>0.2464</td><td>0.9231</td><td>0.9130</td><td>0.9183</td><td>0.9086</td><td><a href="./notebooks/MushroomA/ResNet50_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.4583</td><td>46.0511</td><td>685.7385</td><td>0.0922</td><td>0.9817</td><td>0.9792</td><td>0.9796</td><td>0.9794</td><td><a href="./notebooks/MushroomB/ResNet50_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>ResNeXt50</td><td>32</td><td>224×224</td><td>8.4995</td><td>4.2285</td><td>23.0</td>
      <td>1.9198</td><td>60.6122</td><td>520.8768</td><td>0.1226</td><td>0.9638</td><td>0.9623</td><td>0.9662</td><td>0.9587</td><td><a href="./notebooks/MushroomA/ResNeXt50_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.9219</td><td>60.6907</td><td>520.3269</td><td>0.0525</td><td>0.9900</td><td>0.9889</td><td>0.9896</td><td>0.9887</td><td><a href="./notebooks/MushroomB/ResNeXt50_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>ConvNeXt-T</td><td>32</td><td>224×224</td><td>8.9379</td><td>4.4548</td><td>27.8</td>
      <td>1.7638</td><td>55.6858</td><td>566.9816</td><td>0.1009</td><td>0.9698</td><td>0.9678</td><td>0.9712</td><td>0.9645</td><td><a href="./notebooks/MushroomA/ConvNeXt_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.8026</td><td>56.9235</td><td>554.8174</td><td>0.0516</td><td>0.9900</td><td>0.9890</td><td>0.9900</td><td>0.9884</td><td><a href="./notebooks/MushroomB/ConvNeXt_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>ConvNeXtV2-T</td><td>32</td><td>224×224</td><td>8.9357</td><td>4.4548</td><td>27.9</td>
      <td>2.7397</td><td>86.4968</td><td>365.0030</td><td>0.0975</td><td>0.9698</td><td>0.9673</td><td>0.9715</td><td>0.9634</td><td><a href="./notebooks/MushroomA/ConvNeXtV2_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.6463</td><td>83.5670</td><td>377.8916</td><td>0.0517</td><td>0.9917</td><td>0.9910</td><td>0.9915</td><td>0.9910</td><td><a href="./notebooks/MushroomB/ConvNeXtV2_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>InceptionNeXt-T</td><td>32</td><td>224×224</td><td>8.4013</td><td>4.1864</td><td>25.8</td>
      <td>1.4138</td><td>44.6346</td><td>707.3308</td><td>0.1137</td><td>0.9646</td><td>0.9654</td><td>0.9705</td><td>0.9606</td><td><a href="./notebooks/MushroomA/InceptionNeXt_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.4211</td><td>44.8778</td><td>703.6657</td><td>0.0564</td><td>0.9867</td><td>0.9861</td><td>0.9869</td><td>0.9858</td><td><a href="./notebooks/MushroomB/InceptionNeXt_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>DeiT-S</td><td>32</td><td>224×224</td><td>8.4948</td><td>4.2408</td><td>21.7</td>
      <td>1.2717</td><td>40.1484</td><td>786.6625</td><td>0.1290</td><td>0.9623</td><td>0.9574</td><td>0.9647</td><td>0.9507</td><td><a href="./notebooks/MushroomA/DeiT_Small_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.3315</td><td>42.0481</td><td>751.0820</td><td>0.0519</td><td>0.9900</td><td>0.9883</td><td>0.9885</td><td>0.9887</td><td><a href="./notebooks/MushroomB/DeiT_Small_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td><b>MarisaNeXt-T (Proposed)</b></td><td><b>32</b></td><td><b>224×224</b></td><td><b>8.9379</b></td><td><b>4.4548</b></td><td><b>27.8</b></td>
      <td><b>2.3132</b></td><td><b>73.0312</b></td><td><b>432.3013</b></td><td><b>0.0997</b></td><td><b>0.9729</b></td><td><b>0.9694</b></td><td><b>0.9714</b></td><td><b>0.9675</b></td><td><b><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomA_R1_V1.ipynb">Link</a></b></td>
      <td><b>2.3330</b></td><td><b>73.6735</b></td><td><b>428.6468</b></td><td><b>0.0528</b></td><td><b>0.9917</b></td><td><b>0.9912</b></td><td><b>0.9917</b></td><td><b>0.9910</b></td><td><b><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomB_R1_V1.ipynb">Link</a></b></td>
    </tr>
  </tbody>
</table>
