import XCTest
@testable import PawvisCore

final class DictationActionTests: XCTestCase {
    func testToggleDictationIsPawvisCategoryWithoutArgument() {
        let action = GestureAction(kind: .toggleDictation)
        XCTAssertEqual(action.kind.category, .pawvis)
        XCTAssertFalse(action.kind.needsArgument)
        XCTAssertNil(action.keyChord)
        XCTAssertEqual(action.feedback, "Dictation toggled")
        XCTAssertEqual(action.summary, "Start / stop dictation (types your speech)")
    }

    func testToggleDictationRoundTripsCodable() throws {
        let action = GestureAction(kind: .toggleDictation, argument: "")
        let data = try JSONEncoder().encode(action)
        let decoded = try JSONDecoder().decode(GestureAction.self, from: data)
        XCTAssertEqual(decoded, action)
    }
}