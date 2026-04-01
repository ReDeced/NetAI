#pragma once
#include <cstdint>
#include <string>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ip_icmp.h>
#include <arpa/inet.h>

// Бинарная структура события — фиксированный размер, без выравнивания
#pragma pack(push, 1)
struct NetworkEventMsg {
    double   timestamp;
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t size;
    uint8_t  protocol;  // 0=TCP 1=UDP 2=ICMP 3=OTHER
};
#pragma pack(pop)

enum class Protocol : uint8_t {
    TCP   = 0,
    UDP   = 1,
    ICMP  = 2,
    OTHER = 3
};

inline bool parse_packet(
    const uint8_t* data,
    uint32_t caplen,
    double timestamp,
    NetworkEventMsg& out
) {
    // Пропускаем Ethernet-заголовок (14 байт)
    constexpr int ETH_HEADER_LEN = 14;
    if (caplen < ETH_HEADER_LEN + sizeof(ip)) return false;

    const ip* iph = reinterpret_cast<const ip*>(data + ETH_HEADER_LEN);

    if (iph->ip_v != 4) return false;  // только IPv4

    uint32_t ip_header_len = iph->ip_hl * 4;
    uint32_t ip_offset = ETH_HEADER_LEN + ip_header_len;

    out.timestamp = timestamp;
    out.src_ip    = iph->ip_src.s_addr;
    out.dst_ip    = iph->ip_dst.s_addr;
    out.src_port  = 0;
    out.dst_port  = 0;
    out.size      = static_cast<uint16_t>(caplen);

    switch (iph->ip_p) {
        case IPPROTO_TCP: {
            if (caplen < ip_offset + sizeof(tcphdr)) return false;
            const tcphdr* th = reinterpret_cast<const tcphdr*>(data + ip_offset);
            out.src_port = ntohs(th->th_sport);
            out.dst_port = ntohs(th->th_dport);
            out.protocol = static_cast<uint8_t>(Protocol::TCP);
            break;
        }
        case IPPROTO_UDP: {
            if (caplen < ip_offset + sizeof(udphdr)) return false;
            const udphdr* uh = reinterpret_cast<const udphdr*>(data + ip_offset);
            out.src_port = ntohs(uh->uh_sport);
            out.dst_port = ntohs(uh->uh_dport);
            out.protocol = static_cast<uint8_t>(Protocol::UDP);
            break;
        }
        case IPPROTO_ICMP:
            out.protocol = static_cast<uint8_t>(Protocol::ICMP);
            break;
        default:
            out.protocol = static_cast<uint8_t>(Protocol::OTHER);
            break;
    }

    return true;
}
