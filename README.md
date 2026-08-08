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
      <td>None</td><td>ConvNeXt-Tiny</td><td>32</td><td>224×224</td><td>8.938</td><td>4.455</td><td>27.8</td>
      <td>1.76</td><td>55.69</td><td>566.98</td><td>0.1009</td><td>0.9698</td><td>0.9678</td><td>0.9712</td><td>0.9645</td><td><a href="./notebooks/MushroomA/ConvNeXt_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.80</td><td>56.92</td><td>554.82</td><td>0.0516</td><td>0.9900</td><td>0.9890</td><td>0.9900</td><td>0.9884</td><td><a href="./notebooks/MushroomB/ConvNeXt_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CBAM</td><td>Model-A1</td><td>32</td><td>224×224</td><td>8.939</td><td>4.456</td><td>28.0</td>
      <td>1.79</td><td>56.59</td><td>557.90</td><td>0.1159</td><td>0.9691</td><td>0.9670</td><td>0.9690</td><td>0.9651</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_A1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.77</td><td>55.96</td><td>564.32</td><td>0.0537</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_A1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CBAM</td><td>Model-A2</td><td>32</td><td>224×224</td><td>8.945</td><td>4.458</td><td>29.4</td>
      <td>2.03</td><td>64.22</td><td>491.61</td><td>0.1104</td><td>0.9713</td><td>0.9689</td><td>0.9691</td><td>0.9687</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_A2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.98</td><td>62.52</td><td>505.11</td><td>0.0517</td><td>0.9883</td><td>0.9863</td><td>0.9869</td><td>0.9862</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_A2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CBAM</td><td>Model-A3</td><td>32</td><td>224×224</td><td>8.939</td><td>4.456</td><td>28.0</td>
      <td>1.89</td><td>59.55</td><td>530.20</td><td>0.1072</td><td>0.9698</td><td>0.9687</td><td>0.9732</td><td>0.9644</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_A3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.77</td><td>55.84</td><td>565.53</td><td>0.0566</td><td>0.9900</td><td>0.9892</td><td>0.9898</td><td>0.9892</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_A3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SE</td><td>Model-B1</td><td>32</td><td>224×224</td><td>8.939</td><td>4.455</td><td>28.0</td>
      <td>1.81</td><td>57.21</td><td>551.92</td><td>0.1158</td><td>0.9683</td><td>0.9683</td><td>0.9718</td><td>0.9650</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_B1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.76</td><td>55.64</td><td>567.58</td><td>0.0560</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_B1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SE</td><td>Model-B2</td><td>32</td><td>224×224</td><td>8.943</td><td>4.456</td><td>29.4</td>
      <td>1.78</td><td>56.05</td><td>563.28</td><td>0.1067</td><td>0.9698</td><td>0.9672</td><td>0.9679</td><td>0.9667</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_B2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.86</td><td>58.86</td><td>536.49</td><td>0.0512</td><td>0.9900</td><td>0.9893</td><td>0.9897</td><td>0.9893</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_B2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SE</td><td>Model-B3</td><td>32</td><td>224×224</td><td>8.939</td><td>4.455</td><td>28.0</td>
      <td>1.69</td><td>53.49</td><td>590.31</td><td>0.0990</td><td>0.9698</td><td>0.9678</td><td>0.9722</td><td>0.9638</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_B3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.72</td><td>54.40</td><td>580.55</td><td>0.0540</td><td>0.9900</td><td>0.9889</td><td>0.9896</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_B3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CA</td><td>Model-C1</td><td>32</td><td>224×224</td><td>8.946</td><td>4.459</td><td>28.0</td>
      <td>1.85</td><td>58.45</td><td>540.16</td><td>0.1295</td><td>0.9638</td><td>0.9617</td><td>0.9650</td><td>0.9586</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_C1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.82</td><td>57.45</td><td>549.71</td><td>0.0601</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_C1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CA</td><td>Model-C2</td><td>32</td><td>224×224</td><td>9.000</td><td>4.486</td><td>30.2</td>
      <td>2.22</td><td>70.12</td><td>450.24</td><td>0.1151</td><td>0.9698</td><td>0.9694</td><td>0.9709</td><td>0.9679</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_C2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.24</td><td>70.81</td><td>445.98</td><td>0.0522</td><td>0.9883</td><td>0.9868</td><td>0.9879</td><td>0.9862</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_C2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>CA</td><td>Model-C3</td><td>32</td><td>224×224</td><td>8.946</td><td>4.459</td><td>28.0</td>
      <td>1.86</td><td>58.57</td><td>539.10</td><td>0.1104</td><td>0.9683</td><td>0.9660</td><td>0.9698</td><td>0.9625</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_C3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.84</td><td>58.06</td><td>543.93</td><td>0.0568</td><td>0.9883</td><td>0.9874</td><td>0.9884</td><td>0.9869</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_C3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D1</td><td>32</td><td>224×224</td><td>8.938</td><td>4.455</td><td>27.8</td>
      <td>1.84</td><td>58.02</td><td>544.19</td><td>0.1038</td><td>0.9713</td><td>0.9684</td><td>0.9703</td><td>0.9666</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D1_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.83</td><td>57.63</td><td>547.92</td><td>0.0511</td><td>0.9900</td><td>0.9893</td><td>0.9903</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D1_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D2</td><td>32</td><td>224×224</td><td>8.938</td><td>4.455</td><td>27.8</td>
      <td>2.25</td><td>70.99</td><td>444.76</td><td>0.0976</td><td>0.9751</td><td>0.9730</td><td>0.9757</td><td>0.9704</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D2_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.22</td><td>70.05</td><td>450.78</td><td>0.0558</td><td>0.9900</td><td>0.9889</td><td>0.9896</td><td>0.9887</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D2_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D3</td><td>32</td><td>224×224</td><td>8.938</td><td>4.455</td><td>27.8</td>
      <td>1.81</td><td>57.28</td><td>551.20</td><td>0.1065</td><td>0.9706</td><td>0.9689</td><td>0.9709</td><td>0.9669</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D3_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.83</td><td>57.94</td><td>545.07</td><td>0.0528</td><td>0.9917</td><td>0.9911</td><td>0.9915</td><td>0.9910</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D3_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td><b>SimAM</b></td><td><b>Model-D4</b></td><td><b>32</b></td><td><b>224×224</b></td><td><b>8.938</b></td><td><b>4.455</b></td><td><b>27.8</b></td>
      <td><b>2.31</b></td><td><b>73.03</b></td><td><b>432.30</b></td><td><b>0.0997</b></td><td><b>0.9729</b></td><td><b>0.9694</b></td><td><b>0.9714</b></td><td><b>0.9675</b></td><td><b><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomA_R1_V1.ipynb">Link</a></b></td>
      <td><b>2.33</b></td><td><b>73.67</b></td><td><b>428.65</b></td><td><b>0.0528</b></td><td><b>0.9917</b></td><td><b>0.9912</b></td><td><b>0.9917</b></td><td><b>0.9910</b></td><td><b><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomB_R1_V1.ipynb">Link</a></b></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D5</td><td>32</td><td>224×224</td><td>8.938</td><td>4.455</td><td>27.8</td>
      <td>2.31</td><td>72.78</td><td>433.78</td><td>0.1052</td><td>0.9744</td><td>0.9720</td><td>0.9759</td><td>0.9683</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D5_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.32</td><td>73.20</td><td>431.39</td><td>0.0548</td><td>0.9900</td><td>0.9892</td><td>0.9898</td><td>0.9892</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D5_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>SimAM</td><td>Model-D6</td><td>32</td><td>224×224</td><td>8.938</td><td>4.455</td><td>27.8</td>
      <td>2.39</td><td>75.36</td><td>418.95</td><td>0.1083</td><td>0.9691</td><td>0.9649</td><td>0.9674</td><td>0.9625</td><td><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D6_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.41</td><td>76.09</td><td>415.01</td><td>0.0517</td><td>0.9900</td><td>0.9892</td><td>0.9898</td><td>0.9892</td><td><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D6_224_MushroomB_R1_V1.ipynb">Link</a></td>
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
      <td>ResNet50</td><td>32</td><td>224×224</td><td>8.207</td><td>4.087</td><td>23.5</td>
      <td>1.45</td><td>45.73</td><td>690.43</td><td>0.2464</td><td>0.9231</td><td>0.9130</td><td>0.9183</td><td>0.9086</td><td><a href="./notebooks/MushroomA/ResNet50_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.46</td><td>46.05</td><td>685.74</td><td>0.0922</td><td>0.9817</td><td>0.9792</td><td>0.9796</td><td>0.9794</td><td><a href="./notebooks/MushroomB/ResNet50_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>ResNeXt50</td><td>32</td><td>224×224</td><td>8.500</td><td>4.229</td><td>23.0</td>
      <td>1.92</td><td>60.61</td><td>520.88</td><td>0.1226</td><td>0.9638</td><td>0.9623</td><td>0.9662</td><td>0.9587</td><td><a href="./notebooks/MushroomA/ResNeXt50_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.92</td><td>60.69</td><td>520.33</td><td>0.0525</td><td>0.9900</td><td>0.9889</td><td>0.9896</td><td>0.9887</td><td><a href="./notebooks/MushroomB/ResNeXt50_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>ConvNeXt-T</td><td>32</td><td>224×224</td><td>8.938</td><td>4.455</td><td>27.8</td>
      <td>1.76</td><td>55.69</td><td>566.98</td><td>0.1009</td><td>0.9698</td><td>0.9678</td><td>0.9712</td><td>0.9645</td><td><a href="./notebooks/MushroomA/ConvNeXt_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.80</td><td>56.92</td><td>554.82</td><td>0.0516</td><td>0.9900</td><td>0.9890</td><td>0.9900</td><td>0.9884</td><td><a href="./notebooks/MushroomB/ConvNeXt_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>ConvNeXtV2-T</td><td>32</td><td>224×224</td><td>8.936</td><td>4.455</td><td>27.9</td>
      <td>2.74</td><td>86.50</td><td>365.00</td><td>0.0975</td><td>0.9698</td><td>0.9673</td><td>0.9715</td><td>0.9634</td><td><a href="./notebooks/MushroomA/ConvNeXtV2_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>2.65</td><td>83.57</td><td>377.89</td><td>0.0517</td><td>0.9917</td><td>0.9910</td><td>0.9915</td><td>0.9910</td><td><a href="./notebooks/MushroomB/ConvNeXtV2_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>InceptionNeXt-T</td><td>32</td><td>224×224</td><td>8.401</td><td>4.186</td><td>25.8</td>
      <td>1.41</td><td>44.63</td><td>707.33</td><td>0.1137</td><td>0.9646</td><td>0.9654</td><td>0.9705</td><td>0.9606</td><td><a href="./notebooks/MushroomA/InceptionNeXt_Tiny_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.42</td><td>44.88</td><td>703.67</td><td>0.0564</td><td>0.9867</td><td>0.9861</td><td>0.9869</td><td>0.9858</td><td><a href="./notebooks/MushroomB/InceptionNeXt_Tiny_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td>DeiT-S</td><td>32</td><td>224×224</td><td>8.495</td><td>4.241</td><td>21.7</td>
      <td>1.27</td><td>40.15</td><td>786.66</td><td>0.1290</td><td>0.9623</td><td>0.9574</td><td>0.9647</td><td>0.9507</td><td><a href="./notebooks/MushroomA/DeiT_Small_224_MushroomA_R1_V1.ipynb">Link</a></td>
      <td>1.33</td><td>42.05</td><td>751.08</td><td>0.0519</td><td>0.9900</td><td>0.9883</td><td>0.9885</td><td>0.9887</td><td><a href="./notebooks/MushroomB/DeiT_Small_224_MushroomB_R1_V1.ipynb">Link</a></td>
    </tr>
    <tr>
      <td><b>MarisaNeXt-T (Proposed)</b></td><td><b>32</b></td><td><b>224×224</b></td><td><b>8.938</b></td><td><b>4.455</b></td><td><b>27.8</b></td>
      <td><b>2.31</b></td><td><b>73.03</b></td><td><b>432.30</b></td><td><b>0.0997</b></td><td><b>0.9729</b></td><td><b>0.9694</b></td><td><b>0.9714</b></td><td><b>0.9675</b></td><td><b><a href="./notebooks/MushroomA/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomA_R1_V1.ipynb">Link</a></b></td>
      <td><b>2.33</b></td><td><b>73.67</b></td><td><b>428.65</b></td><td><b>0.0528</b></td><td><b>0.9917</b></td><td><b>0.9912</b></td><td><b>0.9917</b></td><td><b>0.9910</b></td><td><b><a href="./notebooks/MushroomB/proposedModel/ConvNeXt_Tiny_M_D4_224_MushroomB_R1_V1.ipynb">Link</a></b></td>
    </tr>
  </tbody>
</table>
