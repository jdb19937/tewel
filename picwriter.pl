#!/usr/bin/perl

@ARGV or die "Usage: $0 pic.any\n";
my $fn = shift(@ARGV);

if (grep $fn =~ /\.$_$/, qw(avi mkv mp4)) {
  my $r = @ARGV ? $ARGV[0] : 16;

  exec('ffmpeg',
    '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
    '-r', $r,
    '-i', '/dev/stdin',
    '-r', $r,
    '-y', $ARGV[0]
  );
  die "$0: ffmpeg: $!\n";
} else {
  exec('convert', 'ppm:-', $fn);
  die "$0: convert: $!\n";
}
