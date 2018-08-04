import argparse
import logging
import os

import tornado.escape
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.web

from materials_entity_recognition import MatRecognition

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"


class MERHandler(tornado.web.RequestHandler):
    route = r'/MER/recognize'
    version = '2018072300'

    def initialize(self, *args, **kwargs):
        if 'MER_models' not in self.application.settings:
            self.application.settings['MER_models'] = {}
        models = self.application.settings['MER_models']

        pid = os.getpid()
        if pid not in models:
            model = MatRecognition()
            models[pid] = model

        self.model = models[pid]

    def post(self):
        def error_wrong_format():
            self.set_status(400)
            self.write({
                'status': False,
                'message': 'No list of paragraphs is found in request body.'
            })

        def error_no_text():
            self.set_status(400)
            self.write({
                'status': False,
                'message': 'One of the paragraphs does not contain "text" field'
            })

        paragraphs = tornado.escape.json_decode(self.request.body)
        if not isinstance(paragraphs, list):
            return error_wrong_format()

        results = []
        for paragraph in paragraphs:
            if 'text' not in paragraph:
                return error_no_text()
            all_materials, precursors, targets, other_materials = self.model.mat_recognize(paragraph['text'])
            results.append({
                'text': paragraph['text'],
                'all_materials': all_materials,
                'precursors': precursors,
                'targets': targets,
                'other_materials': other_materials
            })

        self.write({
            'status': True,
            'paragraphs': results,
            'version': self.version
        })


def make_app():
    all_handlers = []
    for i in [MERHandler]:
        logging.info('Registering handler %r at %s', i, i.route)
        all_handlers.append([i.route, i, {}])
    return tornado.web.Application(all_handlers)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Synthesis Project Web Service: materials entity recognition.')
    parser.add_argument('--address', action='store', type=str, default='127.0.0.1',
                        help='Listen address.')
    parser.add_argument('--port', action='store', type=int, default=7730,
                        help='Listen port.')
    args = parser.parse_args()

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.bind(address=args.address, port=args.port)
    logging.info('Going to main loop on %s:%d...', args.address, args.port)
    logging.info('Spawning processes...')
    server.start(4)
    try:
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        pass
