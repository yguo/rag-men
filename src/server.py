from http.server import SimpleHTTPRequestHandler
import socketserver
from urllib.parse import parse_qs

from src.pipeline.pipeline import SimplePipeline

PORT = 8777
httpd = None

class ChatServer(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = SimplePipeline()

    def do_POST(self):
       if self.path == "/api/chat":
           content_length = int(self.headers['Content-Length'])
           post_data = self.rfile.read(content_length)
           data = parse_qs(post_data.decode('utf-8'))
           query = data.get('text','')

           self.send_response(200)
           self.send_header("Content-type", "text/plain")
           self.send_header("Access-Control-Allow-Origin", "*") # For CORS
           self.end_headers()
           for chunk in self.pipeline.process_query(query):
               self.wfile.write(chunk.encode('utf-8'))
               self.wfile.flush()
       else:
         self.send_error(404, "Not Found API PATH defined")
           



def run(pipeline):
    global httpd
    handler = lambda *args, **kwargs: ChatServer(pipeline, *args, **kwargs)
    httpd = socketserver.TCPServer(("", PORT), handler)
    print(f"Serving at port {PORT}")
    httpd.serve_forever()


def stop():
    global httpd
    if httpd:
        httpd.shutdown()
        httpd.server_close()
        print(f"Server at port {PORT} stopped. Bye!")

