#!/usr/bin/perl

@ARGV or die "Usage: $0 vid.any\n";
my $fn = shift(@ARGV);

if (my $filter = $ENV{TEWEL_VIDWRITER_FILTER}) {
  open(STDIN, "$filter |") or die "$0: $filter: $!";
}

my $rate = $ENV{TEWEL_VIDWRITER_RATE} || 16;
my $bitrate = $ENV{TEWEL_VIDWRITER_BITRATE};
my $codec = $ENV{TEWEL_VIDWRITER_CODEC};

exec('ffmpeg',
  '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
  '-r', $rate,
  '-i', '/dev/stdin',
  '-r', $rate,
  $bitrate ? ('-b:v', $bitrate) : ( ),
  $codec ? ('-vcodec', $codec) : ( ),
  '-y', $fn
);
die "$0: ffmpeg: $!\n";
