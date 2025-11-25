import threading
import queue
import time
from server.config import *
from onnx_predict import Classifier
from server.logger_ser import logger


class PromiseWorker(threading.Thread):
    def __init__(self, result_pool):
        self.requests = queue.Queue()
        self.result_pool = result_pool
        threading.Thread.__init__(self)

    def return_result(self, uuid, result):

        self.result_pool[uuid] = result

    def run(self):
        self.classifier = Classifier()
        # handle predict
        while True:
            try:
                # get info in queue
                _info_schedule, uuid = self.requests.get(True, 0.05)
                # _info_schedule = self.requests.get(True, 0.05)
                _img_source = _info_schedule["img_source"]
                print("img_source: ", _img_source)
                # _path_result = _info_schedule["path_result"]
                # st = time.time()
                is_valid, result = self.classifier.onnx_classify(_img_source, uuid)
                self.return_result(uuid, (is_valid, result))

            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(e, exc_info=True)
                self.return_result(uuid, False)

    def put_schedule(self, args):
        image_path, uuid = args
        self.requests.put((image_path, uuid))


class Promise(threading.Thread):

    def __init__(self):
        self.requests = queue.Queue()
        self.result_pool = {}

        self.number_workder = NUM_WORKERS
        self.worker_pool = dict()
        self.worker_index = 0
        self.__init_worker()
        threading.Thread.__init__(self)

    def __init_worker(self):
        for i in range(self.number_workder):
            worker = PromiseWorker(self.result_pool)
            self.worker_pool[i] = worker
            worker.start()

    def put_schedule(self, image_path, uuid):
        self.requests.put((image_path, uuid))

    def run(self):
        while True:
            try:
                work = self.requests.get(True, 0.05)
                self.requests.task_done()
                self.worker_pool[self.worker_index].put_schedule(work)
                self.worker_index = (self.worker_index + 1) % self.number_workder
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break

    def get_result(self, uuid):
        while True:
            if uuid in self.result_pool:
                result = self.result_pool.pop(uuid, None)
                return result
            else:
                time.sleep(0.01)

