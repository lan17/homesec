import type { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'com.levneiman.homesec',
  appName: 'HomeSec',
  webDir: 'dist',
  experimental: {
    ios: {
      spm: {
        swiftToolsVersion: '6.2',
      },
    },
  },
}

export default config
