#!/usr/bin/perl

@ARGV or die "Usage: $0 vid.any\n";
my $fn = shift(@ARGV);

if (my $filter = $ENV{TEWEL_VIDWRITER_FILTER}) {
  $filter =~ s#^//#/opt/makemore/share/tewel/#;
  open(STDIN, "$filter |") or die "$0: $filter: $!";
}

my $rate = $ENV{TEWEL_VIDWRITER_RATE} || 16;
my $bitrate = $ENV{TEWEL_VIDWRITER_BITRATE};
my $codec = $ENV{TEWEL_VIDWRITER_CODEC};
my $scale = $ENV{TEWEL_VIDWRITER_SCALE};

exec('ffmpeg',
  '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
  '-r', $rate,
  '-i', '/dev/stdin',
  '-r', $rate,
  $scale ? ('-vf', "scale=$scale") : ( ),
  $bitrate ? ('-b:v', $bitrate) : ( ),
  $codec ? ('-vcodec', $codec) : ( ),
  '-y', $fn
);
die "$0: ffmpeg: $!\n";
