## ML projects on tensorflow

### Environment set up
In the terminal run <b>poetry install</b>, after the installer finishes, run <b>poetry shell</b> and then <b>pip install tensorflow</b> to add tensorflow in the environment

### Projects
<ol>
  <li><h3>Road classification</h3>
    <ul>
      <li><a href="https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification" target="_blank"> link of the dataset</a>, the goal is to create a classifier that detects if a road is clean or not.</li>
      <li>Used mobilenet as feature extractor and for fine tuning</li>
      <li>The script expects a <b>road_classification.zip</b> file in the Downloads home folder</li>
    </ul>
  </li>
  <br>
  <li>
    <h3>Visual pollution</h3>
     <ul>
     	<li><a href="https://www.kaggle.com/datasets/abhranta/urban-visual-pollution-dataset" target="_blank"> link of the dataset</a>, the goal is to create an image detector that finds visual pollutions in an image.</li>
      <li>Used mobilenet as feature extractor and for fine tuning.</li>
      <li>The script expects a <b>visual_pollution.zip</b> file in the Downloads home folder</li>
     </ul>
   </li>
   <br>
  <li>
    <h3>Brain tumors</h3>
     <ul>
     	<li><a href="https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c" target="_blank"> link of the dataset</a>, the goal is to create a classifier that sees a brain scan and decides what brain tumor it is.</li>
      <li>EfficientNet was the best performer for trasnfer learning.</li>
      <li>The script expects a <b>kaggle.json</b> file with the kaggle API credentials in the workding directory of the script to be used by <b>opendatasets</b> to download and extract the dataset in the <b>datasets</b> directory</li>
     </ul>
   </li>
</ol>

