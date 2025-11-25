import sys
import os
import cv2
sys.path.append('.')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from io import BytesIO
from logger_ser import logger
from flask import send_file
from config import *
import uuid
import threading
from promise import Promise
from flask_cors import CORS
from flask import request
from flask import Flask, jsonify, redirect

app = Flask(__name__)
CORS(app)
current_path = os.path.dirname(os.path.abspath(__file__))

root_path = os.path.dirname(current_path)

app.config["status_magic"] = "off"


def __load_cache():
    cache = [s.split(".")[0] for s in os.listdir(PATH_LOG_RESULT)]
    return cache


if not os.path.isdir(PATH_LOG_SOURCE):
    os.makedirs(PATH_LOG_SOURCE)
if not os.path.isdir(PATH_LOG_RESULT):
    os.makedirs(PATH_LOG_RESULT)


def __handle_request(_id, _img):
    try:
        _uuid = uuid.uuid4().hex
        _info_schedule = {"img_source": _img}

        _promise.put_schedule(_info_schedule, _uuid)

        is_valid, result = _promise.get_result(_uuid)
        if not is_valid:
            logger.error("Server false!", exc_info=True)
            return False, None, "Server false!", 1
        else:
            logger.info("Success!")
            return True, result, "success", 0
    except:
        logger.error("Can not save image file in request", exc_info=True)
        return False, None, "Can not save image file in request", 2


"""
@api {post} /api/classifcation/general_document_classification General Document Classification
@apiGroup EKYC
@apiName GeneralDocumentClassification
@apiParam {File} image Image file
@apiParam {String} id ID of image concat with number of component


@apiSuccess {File} _ file after transform
@apiError {String} message "Request Error" or Message of error
@apiError {String} id Id is sent from Client
"""


@app.route('/upload', methods=["GET", "POST"])
def ocr_type_document():
    global cache
    try:
        _id = request.form.get("id")
        if _id is None:
            logger.error("Id not found!")
            raise Exception
        _img_file = request.files["image"]
        print(_img_file)
        is_success, result, message, code = __handle_request(_id, _img_file)
        if is_success:
            result.seek(0)
            return send_file(result, mimetype='image/png'), 200
        else:
            return jsonify({"code": code, "message": message}), 200

    except Exception as e:
        logger.error(e, exc_info=True)
        return jsonify({"code": 3, "message": "Server Error: {}".format(e)}), 200

# @app.route("/", methods=["GET"])
# def index():
#     return redirect("/static/index.html")


cache = __load_cache()
thread_locker = threading.Lock()
_promise = Promise()
_promise.start()

if __name__ == "__main__":
    cache = __load_cache()
    thread_locker = threading.Lock()
    _promise = Promise()
    _promise.start()
    app.run(host="0.0.0.0",port=PORT, debug=True)
