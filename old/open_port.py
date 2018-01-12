import socket

class OpenSSH():
    def __init__(self):
        self.OS = 'MAC'

    def isOpen(self, ip, port):
       s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       try:
          s.connect((ip, int(port)))
          s.shutdown(2)
          return True
       except:
          return False

def main():
    ssh = OpenSSH()
    print(ssh.isOpen('192.168.2.14', 22))

if __name__ == '__main__':
    main()
