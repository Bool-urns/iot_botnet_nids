# IoT-based Network Intrusion Detection

<b>This is my masters project, which won the PwC Ireland award for M.Sc in Computing (March 2020)</b>
<h1>Brief Synopsis</h1>
<p>The project explores the idea of machine learning-based network intrusion detection (NIDS) on so-called 'Internet of Things' (IoT) devices, focusing on one of the biggest threats to IoT devices: becoming compromised as part of a Botnet.</p>
<p>From examining prior research, it was determined that deploying a lightweight NIDS on gateway devices in the IoT architecture could be a viable solution for preventing egde IoT devices from being compromised as part of a botnet. Over the course of this project, a simple NIDS was created on a Raspberry Pi single-board computer and five lightwieght classification algorithms were evaluated for use as part of the system. Each algorithm was trained using simulated botnet attack data, allowing for specific multi-class classification of attack behaviour. The whole system was evaluated in terms of a number of performance metrics and energy consumption (a circuit containing current sensor was created to measure this).</p>






<h1>Project Structure</h1>
<ul>
  <li>The final project paper can be found in the docs folder</li>
  <li>src contains:
    <ol>
      <li><b>Dataset and Features</b>: this contains the condensed dataset used in this project orginally from *here* and Python notebooks explaining the feature extraction process</li>
      <li><b>Measuring</b>: this contains the bash and Python scripts used for measuring measuring the metrics outlined the Metrics section below</li>
      <li><b>NIDS</b>: this contains the full implementation of the basic NIDS used for testing in this project</li>
      <li><b>Classification</b>: this contains the implementations used of the five classification algorithms outlined below</li>
</ul>  

<h2>Installation and Usage</h2>
<p>Due to the fact that this was a research project with a focus on attaining results instead of creating a usable application for others, this page is intended more as a guide to the project itself and is not intended to be installed or used by others.</p>

