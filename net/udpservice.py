# -*- coding: utf-8 -*-
'''服务端（UDP协议局域网广播）'''

import time
import socket

ADDR = ('<broadcast>', 10001)
udpService= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpService.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

while 1:
    udpService.sendto(str('~').encode("utf-8"), ADDR)
    time.sleep(3)