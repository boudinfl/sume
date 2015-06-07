# -*- coding: utf-8 -*-

import multiprocessing


class Worker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, summarizer):
        multiprocessing.Process.__init__(self)
        self.summarizer = summarizer
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            task = self.task_queue.get()
            if task is None:
                self.task_queue.task_done()
                break
            summary, sentence = task
            self.result_queue.put(
                (sentence, self.summarizer.average_cosinus_similarity(
                    summary | {sentence})))
            self.task_queue.task_done()
        return


class Server():

    def __init__(self, summarizer, n_workers=None):
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        if n_workers is None:
            self.n_workers = multiprocessing.cpu_count() * 2
        else:
            self.n_workers = n_workers
        self.workers = [Worker(self.tasks,
                               self.results,
                               summarizer)
                        for i in xrange(self.n_workers)]
        for worker in self.workers:
            worker.start()

    def compute_sims(self, summary, sentences):
        for sentence in sentences:
            self.tasks.put((summary, sentence))
        self.tasks.join()
        results = []
        while not self.results.empty():
            results.append(self.results.get())
        return results

    def exit(self):
        for i in xrange(self.n_workers):
            self.tasks.put(None)
        self.tasks.join()
        self.tasks.close()
        self.results.close()
        return
