# IoT-based Network Intrusion Detection

<b>This is my masters project, which won the PwC Ireland award for M.Sc in Computing (March 2020)</b>
<h1>Brief Synopsis</h1>
<p>The project explores the idea of machine learning-based network intrusion detection (NIDS) on so-called 'Internet of Things' (IoT) devices, focusing on one of the biggest threats to IoT devices: becoming compromised as part of a Botnet.</p>
<p>From examining prior research, it was determined that deploying a lightweight NIDS on gateway devices in the IoT architecture could be a viable solution for preventing egde IoT devices from being compromised as part of a botnet. Over the course of this project, a simple NIDS was created on a Raspberry Pi single-board computer and five lightwieght classification algorithms were evaluated for use as part of the system. Each algorithm was trained using simulated botnet attack data, allowing for specific multi-class classification of attack behaviour. The whole system was evaluated in terms of a number of performance metrics and energy consumption (a circuit containing current sensor was created to measure this).</p>




<h1>Project Structure</h1>
<ul>
  <li>The final project paper can be found in the <b>docs</b> folder</li>
  <li><b>src</b> contains:
    <ol>
      <li><b>Dataset and Features</b>: this contains the condensed dataset used in this project orginally from <a href="https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/bot_iot.php">here</a> and Python notebooks explaining the feature extraction process</li>
      <li><b>Measuring</b>: this contains the bash and Python scripts used for measuring the metrics outlined the Metrics section below</li>
      <li><b>NIDS</b>: this contains the full implementation of the basic NIDS used for testing in this project</li>
      <li><b>Classification</b>: this contains the implementations used of the five classification algorithms outlined below</li>
    </ol>
</ul>  

<h2>Installation and Usage</h2>
<p>Due to the fact that this was a research project with a focus on attaining results instead of creating a usable application for others, this page is intended more as a guide to the project itself and is not intended to be installed or used by others.</p>

<h1>Data Gathering</h1>
  <p>With the huge Machine Learning element to the project, using a high-quality and realistic dataset was fundemental. Initially, the intention with this project was to create a dataset to be used too. However after re-evaluating the priorities of the project, I decided this was outside the scope.</p>
  <p>Many datasets exist for ML-based intrusion detection, like <a href="https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html">KDD</a> and <a href="https://www.ll.mit.edu/r-d/datasets">the DARPA ones</a> however with the focus of the project being on IoT and botnets, these datasets seemed unrealistic (and also quite old). A Number of IoT-based datasets existed such as the one created from <a href="https://arxiv.org/abs/1804.04159">this paper</a> but as <a href="https://arxiv.org/abs/1811.00701">this comparison paper</a> found in it's evaluation of IoT IDS datasets, generally these datatsets suffered from a lack of inclusion of IoT traffic, were not realistic or did not include realistic attack scenarios. This same paper that did these comparisons also created it's own dataset called <a href="https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/bot_iot.php">BoT-IoT</a>.</p>
  <p>This dataset aimed to combat the three issues listed above by simulating more realistic attack scenarios from a range of IoT devices and was ultimately chosen to be used in this project.</p>
<h1>Dataset Evaluation and Feature Selection</h1>
<p>The dataset contained ten attack classes, broken up into three main groups:</p>
<ul>
  <li><b>Denial of service</b>:</li>
  <ul>
    <li>TCP Dos</li>
    <li>TCP DDoS</li>
    <li>UDP DoS</li>
    <li>UDP DDoS</li>
    <li>HTTP DoS</li>
    <li>HTTP DDos</li>
  </ul>  
  <li><b>Data theft</b>:</li>
  <ul>
    <li>Key logging</li>
    <li>Data Exfiltration</li>
  </ul>
  <li><b>Data gathering</b>:</li>
  <ul>
    <li>Service Scanning</li>
    <li>OS fingerprinting</li>
</ul>
  <p></p>
<p>This meant that the models would be trained using a multi-class dataset of eleven classes (including normal/benign traffic). However
<h1>The Five Algorithms</h1>
<p>The five algorithms below were used for comparison:</p>  
<ol>
  <li><a href="https://github.com/Microsoft/EdgeML/wiki/ProtoNN">ProtoNN</a></li>
  <li><a href="https://github.com/BonsaiAI">Bonsai</a></li>
  <li><a href="https://github.com/Microsoft/EdgeML/wiki/ProtoNN">ProtoNN</a></li>
  <li><a href="https://xgboost.readthedocs.io/en/latest/">XGBoost</a></li>
  <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest (Scikit learn implementation)</a></li>
</ol>  
<h1>System Design and Implementation</h1>
<h1>Metrics</h1>
<h1>Results</h1>
