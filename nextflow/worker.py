from flask import Flask, request, send_file, jsonify
from logging.handlers import TimedRotatingFileHandler
import subprocess
import os
import shutil
import datetime
from FilesUtilities import compressFiles
import logging
import json

app = Flask(__name__)



logPath = '/workdir/logs/'



if not os.path.exists(logPath):
    os.makedirs(logPath)



# format the log entries 
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler = TimedRotatingFileHandler(logPath + 'cofactor.log', when='midnight', backupCount=20)
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


PROCESS_PATH_MEMOTE = '/home/main.nf'
SUBMISSIONS_PATH = '/workdir/workerSubmissions/'
RESULTS_PATH = '/workdir/resultsWorker/'



if not os.path.exists(SUBMISSIONS_PATH):
    os.makedirs(SUBMISSIONS_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)



start_time = datetime.datetime.now()



running = False



logger.info("Cofactor Worker running...")





@app.route("/start", methods=["POST"])
def startCofactor():



    global start_time
    global running



    logger.info("Submission in progress...")



    start_time = datetime.datetime.now()



    if os.path.exists(SUBMISSIONS_PATH):
        shutil.rmtree(SUBMISSIONS_PATH, ignore_errors=True)



    if os.path.exists(RESULTS_PATH):
        shutil.rmtree(RESULTS_PATH, ignore_errors=True)



    os.makedirs(SUBMISSIONS_PATH)
    os.makedirs(RESULTS_PATH)



    logger.debug("Directories reset complete")



    try:
        fasta_file, xml_file, json_file = None, None, None
        for file in request.files.values():
            filename = file.filename
            destination = SUBMISSIONS_PATH + str(file.filename)
            logger.info("Saving file: " + destination)
            file.save(destination)
            if filename.endswith('.fasta') or filename.endswith('.fa') or filename.endswith('.faa'):
                fasta_file = destination
            elif filename.endswith('.xml') or filename.endswith('.sbml'):
                xml_file = destination
            elif filename.endswith('.json'):
                json_file = destination

        
        if not (fasta_file):
            logger.error("Missing fasta file")
            return jsonify({"error": "Missing fasta file"}), 400
        
        if not (json_file):
            json_file = '/home/config.json'

        with open(json_file) as f:
            json_data = json.load(f)
        
        prediction_model = json_data.get('prediction_model', 'cnn')


        if not (xml_file):
            p=subprocess.Popen(['nextflow', 'run', '/home/main.nf',
                               '--fasta', fasta_file,
                               '--config' ,json_file,
                               '--prediction_model', prediction_model,
                                '-w', '/workdir/workflow'])
        else:
            p=subprocess.Popen(['nextflow', 'run', '/home/main.nf',
                               '--fasta', fasta_file,
                               '--model', xml_file,
                               '--config' ,json_file,
                                '--prediction_model', prediction_model,
                                '-w', '/workdir/workflow'])

        p.wait()

        logger.info("Nextflow process finished")

        for file in os.listdir('/workdir/workflow'):
            if file.endswith('.tsv') or file.endswith('.xml') or file.endswith('.json'):
                shutil.move(f'/workdir/workflow/{file}', f'{RESULTS_PATH}/{file}')

        shutil.rmtree('/workdir/workflow', ignore_errors=True)

        compressFiles(RESULTS_PATH, RESULTS_PATH + 'results.zip')

        with open(RESULTS_PATH + "/processComplete", "w") as process_complete:
            process_complete.write("Process complete!")



        running = True

        logger.info("returning code 102!")



        return ('processing', 102)



    except IndexError:
        logger.info('An error occurred while processing the submission, returning code 500!')
        return ('error', 500)



@app.route('/status')
def display_msg():



    files = os.listdir(RESULTS_PATH)



    if not running:
        return jsonify({"result": "Not running"}), 410



    if abs(datetime.datetime.now() - start_time) > datetime.timedelta(hours=2):
        return jsonify({"result": "Time out"}), 408



    if "processComplete" in files:



        if "400" in files:
            with open(RESULTS_PATH + "/400", "r") as error_file:
                message = str(error_file.readlines()[0])
            return jsonify({"result": message}), 400



        return send_file(RESULTS_PATH + "results.zip", as_attachment=True,
                        download_name='results.zip'), 200



    return jsonify({"result": "running"}), 202





@app.route('/handshake')
def handshake():



    logger.info("Handshake requested!")



    return jsonify({"result": "alive"}), 200





@app.route('/retrieveLogs')
def retrieveLogs():



    logsPath = '/workdir/logs'
    logsZip = '/logs.zip'



    if os.path.exists(logsZip):
        os.remove(logsZip)



    compressFiles(logsPath, logsZip)



    if not os.path.exists(logsZip):
        return jsonify({"Result": "Request unavailable!"}), 503



    return send_file(logsZip, as_attachment=True, attachment_filename='logs.zip')





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, threaded=False, debug=True)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0