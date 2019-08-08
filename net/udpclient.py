'''客户端（UDP协议局域网广播）'''

import socket

udpClient = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpClient.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
udpClient.bind(('', 10001))
print('Listening for broadcast at ', udpClient.getsockname())

while True:
  data, address = udpClient.recvfrom(65535)
  print('Server received from {}:{}'.format(address, data.decode('utf-8')))