// Copyright 2026 LiveKit, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import Foundation
import OnnxRuntimeBindings

/// One process-wide ``ORTEnv`` and session options for an ``ExecutionProvider``.
enum ORTRuntime {
    private static let lock = NSLock()
    private static var cachedEnv: ORTEnv?

    static func sharedEnv() throws -> ORTEnv {
        lock.lock()
        defer { lock.unlock() }
        if let env = cachedEnv { return env }
        let env = try wrap { try ORTEnv(loggingLevel: .warning) }
        cachedEnv = env
        return env
    }

    static func sessionOptions(for provider: ExecutionProvider) throws -> ORTSessionOptions {
        let options = try wrap { try ORTSessionOptions() }
        if provider == .cpu { return options }
        let coreml = ORTCoreMLExecutionProviderOptions()
        coreml.useCPUAndGPU = provider == .coreMLCPUAndGPU
        coreml.useCPUOnly = provider == .coreMLCPUOnly
        try wrap { try options.appendCoreMLExecutionProvider(with: coreml) }
        return options
    }

    static func wrap<T>(_ body: () throws -> T) throws -> T {
        do {
            return try body()
        } catch let error as WakeWordError {
            throw error
        } catch {
            throw WakeWordError.runtimeFailure(underlying: error)
        }
    }
}
